from book2audio.config import CHARS_PER_SEC
from book2audio.output.audio import expected_secs, plausible
from book2audio.output.m4b import ffmeta_escape


def test_ffmeta_escape():
    assert ffmeta_escape("a=b;c#d\\e\nf") == "a\\=b\\;c\\#d\\\\e\\\nf"
    assert ffmeta_escape(None) == ""


def test_expected_secs():
    assert expected_secs("а") == 0.6                       # floor
    assert expected_secs("а" * 130) == 130 / CHARS_PER_SEC


def test_plausible():
    text = "а" * 130                                       # ~10 s expected
    assert plausible(10.0, text)
    assert not plausible(1.0, text)
    assert not plausible(30.0, text)
