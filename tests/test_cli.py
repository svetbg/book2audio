import sys

import pytest

from book2audio.__main__ import main
from book2audio.args import parse_arguments


def test_missing_file_exits_1_with_error(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["book2audio", str(tmp_path / "nope.txt")])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    assert "\nERROR: File not found" in capsys.readouterr().err


def test_unsupported_format_exits_1(tmp_path, monkeypatch, capsys):
    f = tmp_path / "b.pdf"
    f.write_text("x")
    monkeypatch.setattr(sys, "argv", ["book2audio", str(f)])
    with pytest.raises(SystemExit) as e:
        main()
    assert e.value.code == 1
    assert "\nERROR: Unsupported format '.pdf'" in capsys.readouterr().err


def test_argument_defaults(monkeypatch):
    monkeypatch.setattr(sys, "argv", ["book2audio", "b.epub"])
    a = vars(parse_arguments())
    assert a == dict(
        ebook="b.epub", output=None, ref=None, ref_text=None, ref_max_secs=20, voice=None,
        design_model="k2-fsa/OmniVoice", narrator=None, save_voice=None, seed=1234,
        model="preslaff/omnivoice-bg", server=None, device="cpu", dtype="auto", no_miopen=False,
        threads=None, gen_kwargs="{}", retries=2, chapters=None, max_chars=250, min_chars=200,
        redo_chapters=None, lexicon=None, stress_dict=None, bitrate="64k", workdir=None, keep_chunks=False,
        list=False, preview=None, design_only=False)
