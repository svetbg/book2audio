import json
import sys

import numpy as np
import pytest
import soundfile as sf

from book2audio.app import Book2Audio
from book2audio.args import parse_arguments
from book2audio.config import DESIGN_SENTENCE
from book2audio.exceptions import InputError
from book2audio.tts.voice import VoiceLibrary
from stubs import needs_ffmpeg, serve, stub_modules

pytestmark = needs_ffmpeg

BODY = "Първо изречение тук. Второ изречение там."
REF_TEXT = "Здравей, това е проба."
DESIGN_MODEL, BASE_MODEL = "k2-fsa/OmniVoice", "preslaff/omnivoice-bg"


@pytest.fixture
def calls(tmp_path, monkeypatch):
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path / "xdg"))
    (tmp_path / "b.txt").write_text(
        "\n\n".join(f"Глава {i}\n\n{BODY}" for i in (1, 2, 3)), encoding="utf-8")
    return stub_modules(monkeypatch)


def run(tmp_path, monkeypatch, *opts):
    monkeypatch.setattr(sys, "argv", ["book2audio", str(tmp_path / "b.txt"),
                                      "--workdir", str(tmp_path / "w"),
                                      "--output", str(tmp_path / "o.m4b"), "--min-chars", "1", *opts])
    Book2Audio(parse_arguments()).run()
    return tmp_path / "w"


def sample(path):
    sf.write(path, np.zeros(3 * 24000, dtype=np.float32), 24000)
    return path


def book_texts(calls):
    return [g["text"] for g in calls.generates if g["text"] != DESIGN_SENTENCE]


def settings(work):
    return json.loads((work / "voice_settings.json").read_text(encoding="utf-8"))


def test_ref_takes_the_sample_path(tmp_path, monkeypatch, calls):
    src = sample(tmp_path / "s.wav")
    monkeypatch.chdir(tmp_path)
    work = run(tmp_path, monkeypatch, "--ref", "s.wav", "--ref-text", REF_TEXT,
               "--ref-max-secs", "10")
    assert (work / "voice_key.txt").read_text(encoding="utf-8") == \
        f"ref:{src.resolve()}:{src.stat().st_mtime}:{REF_TEXT}:1234:10.0"   # --ref-max-secs is a float
    assert not (work / "designed_voice.wav").exists()
    assert {g["ref_audio"] for g in calls.generates} == {str(work / "reference.wav")}
    assert settings(work)["ref"] == str(src.resolve())          # a relative --ref is stored absolute
    assert settings(work)["ref_max_secs"] == 10


def test_narrator_uses_the_saved_voice(tmp_path, monkeypatch, calls):
    VoiceLibrary().save("bob", {"ref_audio": str(sample(tmp_path / "s.wav")), "ref_text": REF_TEXT})
    saved = tmp_path / "xdg" / "book2audio" / "voices" / "bob" / "reference.wav"
    work = run(tmp_path, monkeypatch, "--narrator", "bob")
    assert settings(work) == dict(ref=str(saved), ref_text=REF_TEXT, voice=None, seed=1234,
                                  design_model=DESIGN_MODEL, ref_max_secs=20)
    assert (work / "voice_key.txt").read_text(encoding="utf-8") == \
        f"ref:{saved}:{saved.stat().st_mtime}:{REF_TEXT}:1234:20"


def test_ref_and_narrator_conflict(tmp_path, monkeypatch, calls):
    with pytest.raises(InputError, match=r"Use either --ref or --narrator, not both\."):
        run(tmp_path, monkeypatch, "--ref", "x.wav", "--narrator", "bob")


def test_designed_run_saves_voice_settings(tmp_path, monkeypatch, calls):
    work = run(tmp_path, monkeypatch)
    assert settings(work) == dict(ref=None, ref_text=None, voice=None, seed=1234,
                                  design_model=DESIGN_MODEL, ref_max_secs=20)
    assert (work / "voice_key.txt").read_text(encoding="utf-8").startswith("designed:None:1234:")


def test_save_voice_stores_the_designed_voice(tmp_path, monkeypatch, calls):
    run(tmp_path, monkeypatch, "--save-voice", "nv", "--design-only")
    saved = tmp_path / "xdg" / "book2audio" / "voices" / "nv"
    assert (saved / "reference.wav").exists()
    assert (saved / "ref_text.txt").read_text(encoding="utf-8") == DESIGN_SENTENCE
    assert not (tmp_path / "o.m4b").exists()


def test_voice_change_discards_finished_chapters(tmp_path, monkeypatch, calls, capsys):
    work = run(tmp_path, monkeypatch)
    assert len(book_texts(calls)) == 6
    run(tmp_path, monkeypatch, "--voice", "female")
    assert "Voice changed: discarding 3 chapter(s)" in capsys.readouterr().out
    assert len(book_texts(calls)) == 12                         # all three chapters rendered again
    assert sorted(p.name for p in work.glob("ch*.wav")) == ["ch001.wav", "ch002.wav", "ch003.wav"]


def test_redo_chapters_deletes_and_renders_again(tmp_path, monkeypatch, calls, capsys):
    run(tmp_path, monkeypatch)
    before = len(book_texts(calls))
    run(tmp_path, monkeypatch, "--redo-chapters", "2")
    assert "Chapter 2 will be regenerated." in capsys.readouterr().out
    assert book_texts(calls)[before:] == ["Глава 2", BODY]


def test_design_model_differs_from_the_base_model(tmp_path, monkeypatch, calls):
    sys.modules["torch"].cuda.is_available = lambda: True
    monkeypatch.setenv("MIOPEN_LOG_LEVEL", "3")             # --device cuda sets these
    monkeypatch.setenv("MIOPEN_FIND_MODE", "FAST")
    run(tmp_path, monkeypatch, "--device", "cuda", "--dtype", "float32",
        "--gen-kwargs", '{"num_step": 4}')
    assert [name for name, _ in calls.loads] == [BASE_MODEL, DESIGN_MODEL]
    assert calls.loads[1][1] == calls.loads[0][1] == dict(device_map="cuda", dtype="f32")
    design = [g for g in calls.generates if g["text"] == DESIGN_SENTENCE]
    assert [g["kw"] for g in design] == [{}]
    assert all(g["kw"] == {"num_step": 4} for g in calls.generates if g not in design)


def test_design_model_equal_to_the_base_model_is_not_loaded_again(tmp_path, monkeypatch, calls):
    run(tmp_path, monkeypatch, "--design-model", BASE_MODEL)
    assert [name for name, _ in calls.loads] == [BASE_MODEL]


def test_preview_writes_preview_wav(tmp_path, monkeypatch, calls):
    work = run(tmp_path, monkeypatch, "--preview", "1")
    assert (work / "preview.wav").exists()
    assert book_texts(calls) == ["Глава 1"]                     # one chunk only
    assert not list(work.glob("ch*.wav"))


def test_missing_chapters_are_reported(tmp_path, monkeypatch, calls, capsys):
    run(tmp_path, monkeypatch, "--chapters", "1")
    assert ("Note: 1 of 3 chapters are done and included. Not yet done: 2-3. "
            "Rerun to continue (done chapters are reused).") in capsys.readouterr().out


def snapshot(work):
    """The contents of the work folder's files, except concat.txt, which holds the folder's own path."""
    return {p.name: p.read_bytes() for p in work.iterdir() if p.is_file() and p.name != "concat.txt"}


def test_server_run_matches_a_local_run_and_resumes_locally(tmp_path, monkeypatch, calls, capsys):
    sample(tmp_path / "s.wav")
    monkeypatch.chdir(tmp_path)
    ref = ("--ref", "s.wav", "--ref-text", REF_TEXT, "--ref-max-secs", "10")
    local, remote = tmp_path / "wl", tmp_path / "wr"
    run(tmp_path, monkeypatch, *ref, "--workdir", str(local), "--output", str(tmp_path / "ol.m4b"))
    local_gens = len(calls.generates)
    with serve(BASE_MODEL, DESIGN_MODEL) as (server, url):
        loads = len(calls.loads)
        run(tmp_path, monkeypatch, *ref, "--server", url, "--workdir", str(remote),
            "--output", str(tmp_path / "or.m4b"))
        assert len(calls.loads) == loads                        # the client loaded no model
    assert f"Using book2audio server at {url} ({BASE_MODEL})\n" in capsys.readouterr().out
    assert len(calls.generates) == 2 * local_gens               # the server did the narration
    assert snapshot(remote) == snapshot(local)
    assert sorted(p.name for p in remote.iterdir()) == sorted(p.name for p in local.iterdir())

    calls.generates.clear()
    run(tmp_path, monkeypatch, *ref, "--workdir", str(remote), "--output", str(tmp_path / "or.m4b"))
    assert calls.generates == []                                # a finished server run resumes locally


def test_server_run_designs_the_voice_on_the_server_and_ignores_engine_options(
        tmp_path, monkeypatch, calls, capsys):
    with serve(BASE_MODEL, DESIGN_MODEL) as (server, url):
        work = run(tmp_path, monkeypatch, "--server", url, "--device", "cuda", "--design-only")
    out = capsys.readouterr().out
    assert "Note: --device/--dtype/--no-miopen/--threads are set on the server, ignoring them here." in out
    assert f"Using book2audio server at {url} ({DESIGN_MODEL})" in out    # with_model
    assert (work / "designed_voice.wav").exists()
    assert [g["text"] for g in calls.generates] == [DESIGN_SENTENCE]


def test_no_server_no_engine_note(tmp_path, monkeypatch, calls, capsys):
    run(tmp_path, monkeypatch, "--device", "cpu", "--preview", "1")
    assert "set on the server" not in capsys.readouterr().out
