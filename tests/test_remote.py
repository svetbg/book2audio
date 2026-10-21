import hashlib
import json
import socket
import sys
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from book2audio.exceptions import ModelError
from book2audio.tts.remote import RemoteNarrator
from stubs import serve, stub_modules

BG, DESIGN = "preslaff/omnivoice-bg", "k2-fsa/OmniVoice"
TEXT = "а" * 130                                   # ~10 s expected


@pytest.fixture
def calls(monkeypatch):
    return stub_modules(monkeypatch)


@pytest.fixture
def served(calls):
    with serve(BG, DESIGN) as (server, url):
        yield server, url


def ref_file(tmp_path):
    f = tmp_path / "ref.wav"
    sf.write(f, np.zeros(2400, dtype=np.float32), 24000)
    return f


def test_info_gives_each_model_its_own_sample_rate(calls):
    with serve(BG, DESIGN) as (server, url):
        server.narrators[DESIGN].sr = 16000
        assert RemoteNarrator(url, BG).sr == 24000
        assert RemoteNarrator(url, DESIGN).sr == 16000
        with urllib.request.urlopen(url + "/info") as r:
            assert json.load(r) == {"version": "0.1.0", "models": {
                BG: {"sample_rate": 24000}, DESIGN: {"sample_rate": 16000}}}


def test_connecting_logs_one_line(served, capsys):
    RemoteNarrator(served[1], BG)
    assert capsys.readouterr().out == f"Using book2audio server at {served[1]} ({BG})\n"


def test_generate_returns_the_stub_audio_and_passes_the_arguments(served, calls, tmp_path):
    narr = RemoteNarrator(served[1], BG, {"num_step": 16})
    ref = ref_file(tmp_path)
    a = narr.generate("Здравей", 7, ref_audio=str(ref), ref_text="проба.", instruct="female")
    assert a.dtype == np.float32 and len(a) == int(24000 * len("Здравей") / 14)
    [g] = calls.generates
    sha = hashlib.sha256(ref.read_bytes()).hexdigest()
    assert g["text"] == "Здравей" and g["ref_text"] == "проба."
    assert g["kw"] == {"instruct": "female", "num_step": 16}
    assert Path(g["ref_audio"]) == Path(served[0].refs.name) / f"{sha}.wav"
    assert Path(g["ref_audio"]).read_bytes() == ref.read_bytes()


def test_gen_kwargs_do_not_leak_to_the_next_client(served, calls):
    RemoteNarrator(served[1], BG, {"num_step": 16}).generate("a", 1)
    RemoteNarrator(served[1], BG).generate("b", 1)
    assert [g["kw"] for g in calls.generates] == [{"num_step": 16}, {}]


def test_reference_is_uploaded_once_and_again_after_the_server_forgets_it(served, calls, tmp_path):
    server, url = served
    narr, ref = RemoteNarrator(url, BG), str(ref_file(tmp_path))
    for _ in range(3):
        narr.generate("a", 1, ref_audio=ref)
    refs = list(Path(server.refs.name).iterdir())
    assert len(refs) == 1 and refs[0].name.endswith(".wav")
    mtime = refs[0].stat().st_mtime_ns
    narr.generate("a", 1, ref_audio=ref)
    assert refs[0].stat().st_mtime_ns == mtime                         # not written again
    refs[0].unlink()                                                   # the server "restarts"
    assert len(narr.generate("a", 1, ref_audio=ref)) > 0
    assert refs[0].exists() and len(calls.generates) == 5


def test_upload_with_the_wrong_hash_is_refused(served):
    req = urllib.request.Request(f"{served[1]}/refs/{'0' * 64}", b"data", method="PUT")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(req)
    assert e.value.code == 400


def test_unknown_model_lists_the_allowed_ones(served):
    with pytest.raises(ModelError) as e:
        RemoteNarrator(served[1], "evil/model")
    assert BG in str(e.value) and DESIGN in str(e.value)


def test_server_refuses_a_model_it_does_not_serve(calls):
    with serve(BG) as (server, url):
        narr = RemoteNarrator(url, BG)
        narr.model_name = "evil/model"
        with pytest.raises(ModelError, match=f"unknown model 'evil/model'. This server has: {BG}"):
            narr.generate("a", 1)


def test_unreachable_server_is_a_model_error():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    with pytest.raises(ModelError, match=f"Cannot reach the book2audio server at http://127.0.0.1:{port}: "):
        RemoteNarrator(f"http://127.0.0.1:{port}", BG)


def test_a_failing_generate_reaches_the_client_and_the_server_keeps_running(served, calls, monkeypatch):
    narr = RemoteNarrator(served[1], BG)
    model = served[0].narrators[BG].model
    good, failed = model.generate, []

    def flaky(**kw):
        if not failed:
            failed.append(1)
            1 / 0
        return good(**kw)
    monkeypatch.setattr(model, "generate", flaky)
    with pytest.raises(ModelError, match="^ZeroDivisionError: division by zero$"):
        narr.generate("a", 1)
    assert len(narr.generate("a", 1)) > 0


def test_a_book2audio_error_reaches_the_client_as_is(served, monkeypatch):
    narr = RemoteNarrator(served[1], BG)

    def refuse(*a, **kw):
        raise ModelError("no instruct here")
    monkeypatch.setattr(served[0].narrators[BG], "generate", refuse)
    with pytest.raises(ModelError, match="^no instruct here$"):
        narr.generate("a", 1, instruct="x")


def test_synth_chunk_retries_with_another_seed(served, calls, monkeypatch):
    seeds, secs = [], [1.0, 10.0]
    sys.modules["torch"].manual_seed = seeds.append
    model = served[0].narrators[BG].model
    monkeypatch.setattr(model, "generate",
                        lambda **kw: np.full(int(secs.pop(0) * 24000), 0.3, dtype=np.float32))
    a = RemoteNarrator(served[1], BG).synth_chunk(TEXT, {}, 5, 2)
    assert len(a) > 9 * 24000 and seeds == [5, 1005]
