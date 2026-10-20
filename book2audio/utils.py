import re


def clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")            # soft hyphens
    s = s.replace("\u00a0", " ")           # non-breaking spaces
    s = re.sub(r"\s+", " ", s)             # keeps U+0301 stress marks intact
    return s.strip()


def has_speech(s: str) -> bool:
    return re.search(r"[^\W_]", s) is not None


def fill_missing_titles(chapters):
    for i, ch in enumerate(chapters, 1):
        if not ch.title:
            first = ch.paragraphs[0] if ch.paragraphs else ""
            ch.title = first if 0 < len(first) <= 80 else f"Глава {i}"


def compact_ranges(nums):
    """[1,2,3,5,7,8] -> '1-3, 5, 7-8'"""
    out, start, prev = [], None, None
    for n in nums + [None]:
        if start is None:
            start = prev = n
        elif n is not None and n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = n
    return ", ".join(out)


def parse_chapters(spec, n):
    if not spec:
        return list(range(1, n + 1))
    sel = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            sel.update(range(int(a) if a else 1, (int(b) if b else n) + 1))
        else:
            sel.add(int(part))
    return sorted(i for i in sel if 1 <= i <= n)
