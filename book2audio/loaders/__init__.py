from pathlib import Path

from book2audio import Book
from book2audio.exceptions import UnsupportedFormat
from book2audio.loaders.epub import load_epub
from book2audio.loaders.txt import load_txt


def load_book(path: Path, min_chars: int) -> Book:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return load_epub(path, min_chars)
    if suffix in (".txt", ".text"):
        return load_txt(path, min_chars)
    raise UnsupportedFormat(f"Unsupported format '{suffix}'. Convert it to EPUB first, e.g. with Calibre:\n"
                            f"  ebook-convert \"{path}\" \"{path.with_suffix('.epub')}\"")
