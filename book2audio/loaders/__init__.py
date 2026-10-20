from pathlib import Path

from book2audio.exceptions import UnsupportedFormat
from book2audio.loaders.base import Loader
from book2audio.loaders.epub import EpubLoader
from book2audio.loaders.txt import TxtLoader

LOADERS = (EpubLoader, TxtLoader)


def for_path(path: Path, min_chars: int) -> Loader:
    """The loader that handles this file type."""
    suffix = path.suffix.lower()
    for cls in LOADERS:
        if suffix in cls.suffixes:
            return cls(path, min_chars)
    raise UnsupportedFormat(f"Unsupported format '{suffix}'. Convert it to EPUB first, e.g. with Calibre:\n"
                            f"  ebook-convert \"{path}\" \"{path.with_suffix('.epub')}\"")
