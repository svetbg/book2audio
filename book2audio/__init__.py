from dataclasses import dataclass, field
from typing import Optional

__version__ = "0.1.0"


@dataclass
class Chapter:
    title: str
    paragraphs: list = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    chapters: list
    cover: Optional[bytes] = None
    cover_ext: str = "jpg"
