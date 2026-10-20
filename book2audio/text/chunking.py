import re

from book2audio import Chapter
from book2audio.config import PAUSE_CHUNK, PAUSE_PARAGRAPH, PAUSE_SCENE
from book2audio.text.lexicon import Lexicon
from book2audio.utils import has_speech


SENTENCE_RE = re.compile(r"[^.!?…]+(?:[.!?…]+[\"»”“'’)\]]*|$)")


def pack(pieces, max_chars):
    out, cur = [], ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur += " " + p
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def split_long(s, max_chars):
    if len(s) <= max_chars:
        return [s]
    for pat in (r"(?<=[,;:])\s+", r"\s+(?=[—–-]\s)", r"\s+"):
        parts = [x for x in re.split(pat, s) if x]
        if len(parts) > 1:
            out = []
            for part in pack(parts, max_chars):
                out.extend(split_long(part, max_chars) if len(part) > max_chars else [part])
            return out
    return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]


class Chunker:
    """Turns a chapter into a plan of ("tts", text) and ("pause", secs) steps."""

    def __init__(self, max_chars: int, lexicon: Lexicon):
        self.max_chars = max_chars
        self.lexicon = lexicon

    def chunk_paragraph(self, para):
        sentences = [s.strip() for s in SENTENCE_RE.findall(para) if s.strip()]
        pieces = []
        for s in sentences:
            pieces.extend(split_long(s, self.max_chars))
        return pack(pieces, self.max_chars)

    def plan(self, ch: Chapter):
        plan = []
        for para in ch.paragraphs:
            para = self.lexicon.apply(para)
            if not has_speech(para):
                plan.append(("pause", PAUSE_SCENE))
                continue
            chunks = [c for c in self.chunk_paragraph(para) if has_speech(c)]
            for i, c in enumerate(chunks):
                plan.append(("tts", c))
                plan.append(("pause", PAUSE_CHUNK if i < len(chunks) - 1 else PAUSE_PARAGRAPH))
        return plan
