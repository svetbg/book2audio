import json
import sys
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from book2audio.app import Book2Audio
from book2audio.args import parse_arguments
from book2audio.config import DESIGN_SENTENCE
from book2audio.tts.voice import DesignedVoice, SampleVoice
from book2audio.workdir import WorkDir
from stubs import needs_ffmpeg, stub_modules


def test_designed_key(tmp_path):
    wav = tmp_path / "designed_voice.wav"
    wav.write_bytes(b"x")
    (tmp_path / "designed_voice.json").write_text(
        json.dumps({"voice": "deep", "seed": 5, "model": "dm"}))
    voice, key = DesignedVoice("deep", 5, "dm", "m").prepare(None, WorkDir(tmp_path))
    assert key == f"designed:deep:5:{wav.stat().st_mtime}"
    assert voice == {"ref_audio": str(wav), "ref_text": DESIGN_SENTENCE}


@needs_ffmpeg
def test_ref_key(tmp_path):
    src = tmp_path / "s.wav"
    sf.write(src, np.zeros(24000, dtype=np.float32), 24000)
    voice, key = SampleVoice(str(src), "Здравей.", 3, 20).prepare(SimpleNamespace(sr=24000), WorkDir(tmp_path))
    assert key == f"ref:{src.resolve()}:{src.stat().st_mtime}:Здравей.:3:20"
    assert voice["ref_audio"] == str(tmp_path / "reference.wav")


@needs_ffmpeg
def test_run_writes_voice_key(tmp_path, monkeypatch):
    stub_modules(monkeypatch)
    book = tmp_path / "b.txt"
    book.write_text("Първо изречение тук. Второ изречение там.\n" * 3, encoding="utf-8")
    work = tmp_path / "w"
    monkeypatch.setattr(sys, "argv", ["book2audio", str(book), "--workdir", str(work),
                                      "--output", str(tmp_path / "o.m4b"), "--min-chars", "1"])
    Book2Audio(parse_arguments()).run()
    wav = work / "designed_voice.wav"
    assert (work / "voice_key.txt").read_text(encoding="utf-8") == \
        f"designed:None:1234:{wav.stat().st_mtime}"
