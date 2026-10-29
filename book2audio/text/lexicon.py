import re
from pathlib import Path
from typing import Optional


STRESS = "́"                                    # what the model was trained on
WORD_RE = re.compile(r"[Ѐ-ӿ̀́]+")  # Cyrillic word, marks included
VOWELS = set("аеиоуъюя")


def load_stress_dict(path: str) -> dict:
    """bulgarian-accenter's stress-dict.txt (front-coded, sorted). Each line:
    chr(48 + chars shared with the previous word), the rest of the word, ':' or ';',
    then the offsets: offset n puts the stress after the n-th letter ("вятър:2" -> вя́тър).
    Words with several offsets (both stresses are in use) are left out: either guess is wrong
    half the time, so they are for the lexicon to decide."""
    d, prev = {}, ""
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        m = re.fullmatch(r"(.)(.*?)[:;]([\d,]+)", line)
        if not m:
            continue
        prev = prev[:ord(m[1]) - 48] + m[2]
        if m[3].isdigit():
            d[prev] = int(m[3])
    return d


class Lexicon:
    """Pronunciation fixes: a list of (compiled pattern, replacement) rules, then optionally
    a stress mark on every other word found in a stress dictionary."""

    def __init__(self, rules=(), stress=None):
        self.rules = list(rules)
        self.stress = stress or {}

    @classmethod
    def from_file(cls, path: Optional[str], stress_path: Optional[str] = None) -> "Lexicon":
        """Lines: 'word<TAB>replacement' or 'word=replacement'. '#' starts a comment.
        Example (stress mark = U+0301 after the vowel):  бутона=буто́на"""
        rules = []
        stress = load_stress_dict(stress_path) if stress_path else None
        if not path:
            return cls(rules, stress)
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
        return cls(rules, stress)

    def _stress_word(self, m):
        w = m.group(0)
        if "̀" in w or STRESS in w:        # already marked (lexicon or book): leave it
            return w
        low = w.lower()
        off = self.stress.get(low)
        if off is None or sum(c in VOWELS for c in low) < 2 or not 0 < off <= len(w) \
                or low[off - 1] not in VOWELS:
            return w
        return w[:off] + STRESS + w[off:]

    def apply(self, text: str) -> str:
        def repl(m, dst):
            if m.group(0)[:1].isupper() and dst:
                return dst[0].upper() + dst[1:]
            return dst
        for pat, dst in self.rules:
            text = pat.sub(lambda m, d=dst: repl(m, d), text)
        if self.stress:
            text = WORD_RE.sub(self._stress_word, text)
        return text
