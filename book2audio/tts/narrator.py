import inspect

import numpy as np

from book2audio.config import DEFAULT_SR
from book2audio.exceptions import MissingDependency, ModelError
from book2audio.logging import log
from book2audio.output.audio import expected_secs, plausible, trim_silence


class Narrator:
    def __init__(self, model_name, device="cpu", threads=None, gen_kwargs=None, dtype="auto",
                 no_miopen=False):
        try:
            import torch
        except ImportError:
            raise MissingDependency("PyTorch is not installed (see the install steps in the README).")
        self.torch = torch
        self.model_name, self.device, self.threads = model_name, device, threads
        self.dtype, self.no_miopen = dtype, no_miopen
        if threads:
            torch.set_num_threads(threads)
        try:
            from omnivoice import OmniVoice
        except ImportError:
            raise MissingDependency("OmniVoice is not installed. Run: pip install omnivoice")

        if device != "cpu" and not torch.cuda.is_available():
            raise ModelError(f"--device {device} requested, but PyTorch sees no GPU. Is the ROCm build of "
                f"PyTorch installed in this environment? Check with:\n"
                f"  python -c \"import torch; print(torch.version.hip, torch.cuda.is_available())\"")
        if no_miopen:
            torch.backends.cudnn.enabled = False   # on ROCm this switches off MIOpen
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else "float16"
        dtype = getattr(torch, dtype)
        log(f"Loading {model_name} on {device} ({str(dtype).replace('torch.', '')}) ...",
              flush=True)
        model, last_err = None, None
        for kw in (dict(device_map=device, dtype=dtype), dict(device_map=device), {}):
            try:
                model = OmniVoice.from_pretrained(model_name, **kw)
                break
            except TypeError as e:
                last_err = e
            except OSError as e:
                raise ModelError(f"Could not download/load '{model_name}'.\n"
                    f"The model may be gated: open https://huggingface.co/{model_name}, accept the "
                    f"conditions, then run 'hf auth login' (or 'huggingface-cli login').\n{e}")
        if model is None:
            raise last_err
        if hasattr(model, "to"):
            try:
                model = model.to(device)
            except Exception:
                pass
        if hasattr(model, "eval"):
            model.eval()
        self.model = model
        self.sr = int(getattr(model, "sampling_rate", None)
                      or getattr(model, "sample_rate", None) or DEFAULT_SR)
        sig = inspect.signature(model.generate)
        self._params = set(sig.parameters)
        self._var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        self.gen_kwargs = gen_kwargs or {}

    def with_model(self, model_name):
        """Another model loaded with the same engine settings (and no extra generate args)."""
        return Narrator(model_name, self.device, self.threads, {}, self.dtype, self.no_miopen)

    def accepts(self, name):
        return self._var_kw or name in self._params

    def generate(self, text, seed, ref_audio=None, ref_text=None, instruct=None):
        torch = self.torch
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        kw = {"text": text}
        if ref_audio:
            kw["ref_audio"] = ref_audio
        if ref_text and self.accepts("ref_text"):
            kw["ref_text"] = ref_text
        if instruct:
            if not self.accepts("instruct"):
                raise ModelError("This OmniVoice version has no 'instruct' (voice design) parameter. "
                    "Run without --voice, or use --ref with a sample.")
            kw["instruct"] = instruct
        kw.update(self.gen_kwargs)
        with torch.inference_mode():
            out = self.model.generate(**kw)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "detach"):
            out = out.detach().float().cpu().numpy()
        a = np.asarray(out, dtype=np.float32).squeeze()
        if a.ndim > 1:
            a = a.mean(axis=0 if a.shape[0] < a.shape[-1] else 1)
        return a

    def synth_chunk(self, text, voice, seed, retries):
        """Generate one chunk; retry with another seed if the length looks broken."""
        best, best_err = None, None
        for attempt in range(retries + 1):
            a = trim_silence(self.generate(text, seed + attempt * 1000, **voice), self.sr)
            dur = len(a) / self.sr
            if plausible(dur, text):
                return a
            err = abs(np.log(max(dur, 0.05) / expected_secs(text)))
            if best is None or err < best_err:
                best, best_err = a, err
        log(f"    ! suspicious length after {retries + 1} tries, keeping best: {text[:60]}…",
              flush=True)
        return best
