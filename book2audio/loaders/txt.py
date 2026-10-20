import re
from pathlib import Path

from book2audio import Book, Chapter
from book2audio.utils import clean_text, fill_missing_titles


CHAPTER_LINE = re.compile(r"^\s*(глава|част|пролог|епилог|chapter|prologue|epilogue)\b",
                          re.IGNORECASE)


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):     # many Bulgarian .txt files are cp1251
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_txt(path: Path, min_chars: int) -> Book:
    text = read_text_file(path).replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", text)
    if len(blocks) < 3:                     # no blank lines: one paragraph per line
        blocks = text.split("\n")
    chapters, cur = [], Chapter(title="")
    for b in blocks:
        b = clean_text(b)
        if not b:
            continue
        if len(b) <= 80 and CHAPTER_LINE.match(b):
            if cur.paragraphs:
                chapters.append(cur)
            cur = Chapter(title=b, paragraphs=[b])
        else:
            cur.paragraphs.append(b)
    if cur.paragraphs:
        chapters.append(cur)
    chapters = [c for c in chapters if c.chars >= min_chars] or chapters
    fill_missing_titles(chapters)
    return Book(title=path.stem, author="", chapters=chapters)
