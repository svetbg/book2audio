from pathlib import Path

from book2audio import Book


class Loader:
    """Reads one kind of ebook file. Subclasses list the file types they handle in `suffixes`."""

    suffixes: tuple = ()

    def __init__(self, path: Path, min_chars: int):
        self.path = path
        self.min_chars = min_chars

    def load(self) -> Book:
        raise NotImplementedError
