from book2audio import Chapter
from book2audio.config import PAUSE_CHUNK, PAUSE_PARAGRAPH, PAUSE_SCENE
from book2audio.text.chunking import chapter_plan, chunk_paragraph, pack, split_long


def test_sentence_split():
    assert chunk_paragraph("Едно. Две! Три?", 5) == ["Едно.", "Две!", "Три?"]


def test_pack_respects_limit():
    out = pack(["aaa", "bbb", "ccc"], 7)
    assert out == ["aaa bbb", "ccc"]
    assert all(len(c) <= 7 for c in out)


def test_split_long_sentence_on_commas():
    s = "alpha beta, gamma delta, epsilon zeta"
    out = split_long(s, 15)
    assert len(out) > 1 and all(len(c) <= 15 for c in out)
    assert " ".join(out).replace(",", "") .split() == s.replace(",", "").split()


def test_split_long_hard_cut():
    assert split_long("x" * 10, 4) == ["xxxx", "xxxx", "xx"]


def test_chapter_plan_pauses():
    ch = Chapter("t", ["Едно. Две.", "* * *"])
    plan = chapter_plan(ch, 5, [])
    assert plan == [("tts", "Едно."), ("pause", PAUSE_CHUNK),
                    ("tts", "Две."), ("pause", PAUSE_PARAGRAPH),
                    ("pause", PAUSE_SCENE)]
