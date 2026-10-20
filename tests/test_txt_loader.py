from book2audio.loaders import load_book
from book2audio.loaders.txt import load_txt

BODY = "Дълъг абзац с текст. " * 20


def test_chapter_lines_detected(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text(f"Глава 1\n\n{BODY}\n\nГлава 2\n\n{BODY}\n", encoding="utf-8")
    book = load_txt(f, 50)
    assert [c.title for c in book.chapters] == ["Глава 1", "Глава 2"]
    assert book.title == "b"


def test_cp1251_decoding(tmp_path):
    f = tmp_path / "b.txt"
    f.write_bytes(f"Глава 1\n\n{BODY}\n".encode("cp1251"))
    assert load_book(f, 50).chapters[0].title == "Глава 1"


def test_min_chars_filter(tmp_path):
    f = tmp_path / "b.txt"
    f.write_text(f"Глава 1\n\nкратко\n\nГлава 2\n\n{BODY}\n", encoding="utf-8")
    assert [c.title for c in load_txt(f, 100).chapters] == ["Глава 2"]
    # when nothing passes the filter, all chapters are kept
    assert len(load_txt(f, 10 ** 6).chapters) == 2
