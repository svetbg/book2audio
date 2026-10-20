import re
from pathlib import Path
from typing import Optional


class Lexicon:
    """Pronunciation fixes: a list of (compiled pattern, replacement) rules."""

    def __init__(self, rules=()):
        self.rules = list(rules)

    @classmethod
    def from_file(cls, path: Optional[str]) -> "Lexicon":
        """Lines: 'word<TAB>replacement' or 'word=replacement'. '#' starts a comment.
        Example (stress mark = U+0301 after the vowel):  бутона=буто́на"""
        rules = []
        if not path:
            return cls(rules)
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            sep = "\t" if "\t" in line else ("=" if "=" in line else None)
            if not sep:
                continue
            src, dst = (x.strip() for x in line.split(sep, 1))
            if src:
                rules.append((re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE), dst))
        return cls(rules)

    def apply(self, text: str) -> str:
        def repl(m, dst):
            if m.group(0)[:1].isupper() and dst:
                return dst[0].upper() + dst[1:]
            return dst
        for pat, dst in self.rules:
            text = pat.sub(lambda m, d=dst: repl(m, d), text)
        return text
