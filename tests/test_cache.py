import hashlib
from types import SimpleNamespace

import numpy as np
import soundfile as sf

from book2audio import Chapter
from book2audio.output.audio import expected_secs
from book2audio.output.render import render_chapter

SR = 24000
TEXT = "Това е един тестов абзац."


class FakeNarrator:
    sr = SR

    def __init__(self):
        self.calls = []

    def generate(self, text, seed, **voice):
        self.calls.append(text)
        t = np.arange(int(expected_secs(text) * SR)) / SR
        return (0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32)


def render(tmp_path, narr, keep_chunks=True):
    args = SimpleNamespace(seed=1, retries=0, keep_chunks=keep_chunks)
    plan = [("tts", TEXT)]
    return render_chapter(narr, 1, 1, Chapter(title="T", paragraphs=[TEXT]), plan, {}, "vk", args, tmp_path)


def test_chunk_file_named_by_md5_of_key_and_text(tmp_path):
    render(tmp_path, FakeNarrator())
    h = hashlib.md5(("vk|" + TEXT).encode("utf-8")).hexdigest()[:10]
    assert (tmp_path / "ch001_chunks" / f"{h}.wav").exists()
    assert (tmp_path / "ch001.wav").exists()


def test_legacy_chunk_name_reused_without_generate(tmp_path):
    h = hashlib.md5(("vk|" + TEXT).encode("utf-8")).hexdigest()[:10]
    cdir = tmp_path / "ch001_chunks"
    cdir.mkdir()
    sf.write(cdir / f"0001_{h}.wav", np.zeros(SR // 2, dtype=np.float32), SR, subtype="PCM_16")
    narr = FakeNarrator()
    render(tmp_path, narr)
    assert narr.calls == []
    assert (tmp_path / "ch001.wav").exists()


def test_finished_chapter_is_skipped(tmp_path):
    (tmp_path / "ch001.wav").write_bytes(b"x")
    narr = FakeNarrator()
    assert render(tmp_path, narr) == tmp_path / "ch001.wav"
    assert narr.calls == []


def test_chunks_removed_unless_kept(tmp_path):
    render(tmp_path, FakeNarrator(), keep_chunks=False)
    assert not (tmp_path / "ch001_chunks").exists()
