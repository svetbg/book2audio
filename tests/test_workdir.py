from book2audio.workdir import WorkDir


def test_voice_key_round_trip(tmp_path):
    wd = WorkDir(tmp_path / "new" / "work")          # created on demand
    assert wd.read_voice_key() is None
    wd.write_voice_key("k")
    assert (wd.path / "voice_key.txt").read_text(encoding="utf-8") == "k"
    assert wd.read_voice_key() == "k"


def test_file_names_and_finished_chapters(tmp_path):
    wd = WorkDir(tmp_path)
    assert wd.chapter_wav(7) == tmp_path / "ch007.wav"
    assert wd.chapter_tmp(7) == tmp_path / "ch007.tmp.wav"
    assert wd.chunk_dir(7) == tmp_path / "ch007_chunks"
    for name in ("ch002.wav", "ch001.wav", "ch003.tmp.wav", "preview.wav"):
        (tmp_path / name).write_bytes(b"x")
    assert wd.finished_chapters() == [tmp_path / "ch001.wav", tmp_path / "ch002.wav"]


def test_chunk_tmp_and_default_folder_names(tmp_path):
    assert WorkDir.chunk_tmp(tmp_path / "abc.wav") == tmp_path / "abc.tmp.wav"
    assert WorkDir.for_output(tmp_path / "book.m4b").path == tmp_path / "book_work"
