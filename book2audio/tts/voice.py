import json
import os
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import soundfile as sf

from book2audio.config import DESIGN_SENTENCE
from book2audio.exceptions import InputError, VoiceError
from book2audio.logging import log
from book2audio.output.audio import normalize, trim_silence
from book2audio.output.m4b import detect_pauses, run_ffmpeg
from book2audio.workdir import WorkDir


class Voice:
    """The voice of the book. prepare() returns (generate() kwargs, cache key)."""

    def prepare(self, narrator, workdir: WorkDir):
        raise NotImplementedError


@dataclass
class SampleVoice(Voice):
    ref: str
    ref_text: Optional[str]
    seed: int
    ref_max_secs: float

    def prepare(self, narrator, workdir: WorkDir):
        src = Path(self.ref).expanduser()
        if not src.exists():
            raise InputError(f"Reference file not found: {src}")
        full = workdir.reference_full
        run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(narrator.sr), str(full)])
        dur = sf.info(str(full)).duration
        ref_text = (self.ref_text or "").strip()

        if dur > self.ref_max_secs + 0.05:
            pauses = [p for p in detect_pauses(full) if 3.0 < p < self.ref_max_secs]
            hint = (f"  Pauses (good cut points) at: {', '.join(f'{p:.1f}s' for p in pauses)}\n"
                    if pauses else "")
            cut = f"{pauses[-1] + 0.2:.1f}" if pauses else "15"
            if ref_text:
                raise VoiceError(f"The sample is {dur:.1f}s long, but only the first {self.ref_max_secs:g}s "
                    f"are used.\nThen --ref-text contains words that are NOT in the reference audio, "
                    f"and the model speaks those words inside the book.\n"
                    f"Cut the sample at a pause and give only the words spoken up to that point:\n"
                    f"{hint}  ffmpeg -i \"{src}\" -to {cut} -c copy sample_cut.wav\n"
                    f"Or use the whole sample with --ref-max-secs {int(dur) + 1} (slower on CPU).")
            log(f"Note: using only the first {self.ref_max_secs:g}s of the {dur:.1f}s sample.")

        ref = workdir.reference
        run_ffmpeg(["-i", str(full), "-t", str(self.ref_max_secs), str(ref)])

        if ref_text:
            used = min(dur, self.ref_max_secs)
            rate = len(re.findall(r"[^\W_]", ref_text)) / max(used, 0.1)
            if re.search(r"\d", ref_text):
                log("Tip: write numbers in --ref-text as words, exactly as they are spoken "
                      "(e.g. 'хиляда осемстотин шейсет и шеста').")
            if not 8 <= rate <= 20:
                log(f"Warning: --ref-text has ~{rate:.0f} letters per second of audio "
                      f"(normal speech is ~12-16). It probably doesn't match the clip exactly, "
                      f"which can make words from the sample appear in the book.")
            if not re.search(r"[.!?…][\"»”“'’)]*$", ref_text):
                ref_text += "."
        else:
            log("Note: no --ref-text given. Giving the exact transcript of the sample "
                  "usually improves cloning.")
        key = f"ref:{src.resolve()}:{src.stat().st_mtime}:{ref_text}:{self.seed}:{self.ref_max_secs}"
        return {"ref_audio": str(ref), "ref_text": ref_text or None}, key


@dataclass
class DesignedVoice(Voice):
    voice: Optional[str]
    seed: int
    design_model: Optional[str]
    model: str

    def prepare(self, narrator, workdir: WorkDir):
        designed = workdir.designed_wav
        meta_f = workdir.designed_json
        want = {"voice": self.voice, "seed": self.seed, "model": self.design_model or self.model}
        have = None
        if meta_f.exists():
            try:
                have = json.loads(meta_f.read_text(encoding="utf-8"))
            except Exception:
                pass
        if designed.exists() and have != want:
            log("Voice settings changed since designed_voice.wav was made: designing a new one.")
            designed.unlink()
        if not designed.exists():
            # The Bulgarian fine-tune may ignore voice descriptions, so by default the
            # voice is designed with the base OmniVoice model and only then cloned.
            dn = narrator if want["model"] == self.model else narrator.with_model(want["model"])
            log(f"No sample given: designing a voice ({self.voice or 'automatic'}) "
                  f"with {want['model']} ...", flush=True)
            a = dn.generate(DESIGN_SENTENCE, self.seed, instruct=self.voice)
            sf.write(designed, normalize(trim_silence(a, dn.sr)), dn.sr, subtype="PCM_16")
            meta_f.write_text(json.dumps(want, ensure_ascii=False), encoding="utf-8")
            if dn is not narrator:
                del dn
                import gc
                gc.collect()
            log(f"Saved {designed}\n  Listen to it. If you don't like it, rerun with "
                  f"another --seed and/or --voice (it is regenerated automatically).")
        key = f"designed:{self.voice}:{self.seed}:{designed.stat().st_mtime}"
        return {"ref_audio": str(designed), "ref_text": DESIGN_SENTENCE}, key


class VoiceLibrary:
    """Voices saved with --save-voice, one folder each under $XDG_DATA_HOME/book2audio/voices."""

    def __init__(self):
        base = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
        self.root = base / "book2audio" / "voices"

    def get(self, name: str) -> dict:
        d = self.root / name
        ref, txt = d / "reference.wav", d / "ref_text.txt"
        if not ref.exists():
            have = sorted(p.name for p in self.root.glob("*") if (p / "reference.wav").exists())
            raise InputError(f"No saved voice named '{name}'. "
                f"Saved voices: {', '.join(have) if have else 'none yet (use --save-voice NAME)'}")
        voice = {"ref_audio": str(ref),
                 "ref_text": txt.read_text(encoding="utf-8").strip() if txt.exists() else None}
        log(f"Using saved voice '{name}'.")
        return voice

    def save(self, name: str, voice: dict):
        d = self.root / name
        d.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(voice["ref_audio"], d / "reference.wav")
        (d / "ref_text.txt").write_text(voice.get("ref_text") or "", encoding="utf-8")
        log(f"Voice saved as '{name}' ({d}). Next time just use: --narrator {name}")
