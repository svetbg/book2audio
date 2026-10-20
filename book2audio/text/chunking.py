import re

from book2audio import Chapter
from book2audio.config import PAUSE_CHUNK, PAUSE_PARAGRAPH, PAUSE_SCENE
from book2audio.text.lexicon import apply_lexicon
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


def chunk_paragraph(para, max_chars):
    sentences = [s.strip() for s in SENTENCE_RE.findall(para) if s.strip()]
    pieces = []
    for s in sentences:
        pieces.extend(split_long(s, max_chars))
    return pack(pieces, max_chars)


def chapter_plan(ch: Chapter, max_chars: int, lexicon):
    plan = []
    for para in ch.paragraphs:
        para = apply_lexicon(para, lexicon)
        if not has_speech(para):
            plan.append(("pause", PAUSE_SCENE))
            continue
        chunks = [c for c in chunk_paragraph(para, max_chars) if has_speech(c)]
        for i, c in enumerate(chunks):
            plan.append(("tts", c))
            plan.append(("pause", PAUSE_CHUNK if i < len(chunks) - 1 else PAUSE_PARAGRAPH))
    return plan
