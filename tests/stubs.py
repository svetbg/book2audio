import shutil
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

needs_ffmpeg = pytest.mark.skipif(not shutil.which("ffmpeg"), reason="ffmpeg is not installed")


def stub_modules(monkeypatch, gpu=False):
    """Fake torch and omnivoice. Returns the log of model loads and generate() calls."""
    calls = SimpleNamespace(loads=[], generates=[])

    class Ctx:
        def __enter__(self): pass
        def __exit__(self, *a): pass

    torch = types.ModuleType("torch")
    torch.float32, torch.float16 = "f32", "f16"
    torch.set_num_threads = torch.manual_seed = lambda *a: None
    torch.cuda = SimpleNamespace(is_available=lambda: gpu)
    torch.backends = SimpleNamespace(cudnn=SimpleNamespace(enabled=True))
    torch.inference_mode = Ctx

    class Model:
        sampling_rate = 24000
        @classmethod
        def from_pretrained(cls, name, **kw):
            calls.loads.append((name, kw))
            return cls()
        def generate(self, text, ref_audio=None, ref_text=None, **kw):
            calls.generates.append(dict(text=text, ref_audio=ref_audio, ref_text=ref_text, kw=kw))
            return np.sin(np.arange(int(24000 * max(len(text), 1) / 14)) * 0.05).astype(np.float32) * 0.3

    omni = types.ModuleType("omnivoice")
    omni.OmniVoice = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "omnivoice", omni)
    return calls
