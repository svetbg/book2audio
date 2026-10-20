import json
import sys
import types
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from book2audio.__main__ import run
from book2audio.args import parse_arguments
from book2audio.config import DESIGN_SENTENCE
from book2audio.tts.voice import prepare_voice


def test_designed_key(tmp_path):
    wav = tmp_path / "designed_voice.wav"
    wav.write_bytes(b"x")
    args = SimpleNamespace(ref=None, voice="deep", seed=5, design_model="dm", model="m")
    (tmp_path / "designed_voice.json").write_text(
        json.dumps({"voice": "deep", "seed": 5, "model": "dm"}))
    voice, key = prepare_voice(None, args, tmp_path)
    assert key == f"designed:deep:5:{wav.stat().st_mtime}"
    assert voice == {"ref_audio": str(wav), "ref_text": DESIGN_SENTENCE}


def test_ref_key(tmp_path):
    src = tmp_path / "s.wav"
    sf.write(src, np.zeros(24000, dtype=np.float32), 24000)
    args = SimpleNamespace(ref=str(src), ref_text="Здравей.", seed=3, ref_max_secs=20)
    voice, key = prepare_voice(SimpleNamespace(sr=24000), args, tmp_path)
    assert key == f"ref:{src.resolve()}:{src.stat().st_mtime}:Здравей.:3:20"
    assert voice["ref_audio"] == str(tmp_path / "reference.wav")


def _stub_modules(monkeypatch):
    class Ctx:
        def __enter__(self): pass
        def __exit__(self, *a): pass

    torch = types.ModuleType("torch")
    torch.float32 = torch.float16 = "dtype"
    torch.set_num_threads = torch.manual_seed = lambda *a: None
    torch.cuda = SimpleNamespace(is_available=lambda: False)
    torch.backends = SimpleNamespace(cudnn=SimpleNamespace(enabled=True))
    torch.inference_mode = Ctx

    class Model:
        sampling_rate = 24000
        @classmethod
        def from_pretrained(cls, name, **kw): return cls()
        def generate(self, text, ref_audio=None, ref_text=None, **kw):
            return np.sin(np.arange(int(24000 * max(len(text), 1) / 14)) * 0.05).astype(np.float32) * 0.3

    omni = types.ModuleType("omnivoice")
    omni.OmniVoice = Model
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "omnivoice", omni)


def test_run_writes_voice_key(tmp_path, monkeypatch):
    _stub_modules(monkeypatch)
    book = tmp_path / "b.txt"
    book.write_text("Първо изречение тук. Второ изречение там.\n" * 3, encoding="utf-8")
    work = tmp_path / "w"
    monkeypatch.setattr(sys, "argv", ["book2audio", str(book), "--workdir", str(work),
                                      "--output", str(tmp_path / "o.m4b"), "--min-chars", "1"])
    run(parse_arguments())
    wav = work / "designed_voice.wav"
    assert (work / "voice_key.txt").read_text(encoding="utf-8") == \
        f"designed:None:1234:{wav.stat().st_mtime}"
