import hashlib
import json
import socket
import sys
import threading
import urllib.error
import urllib.request
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest
import soundfile as sf

from book2audio.exceptions import ModelError
from book2audio.tts import remote
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


@contextmanager
def fake_server(reply=None):
    """Answers every request with the raw bytes `reply`, or with silence if None. Yields its URL."""
    srv = socket.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen()
    held = []

    def loop():
        while True:
            try:
                c, _ = srv.accept()
            except OSError:
                return
            held.append(c)                      # closed at the end, so a silent server stays silent
            c.recv(65536)
            if reply is not None:
                c.sendall(reply)
                c.shutdown(socket.SHUT_WR)          # FIN, not RST: the request may not be read fully
    threading.Thread(target=loop, daemon=True).start()
    try:
        yield f"http://127.0.0.1:{srv.getsockname()[1]}"
    finally:
        srv.close()
        for c in held:
            c.close()


def http(body, status=b"200 OK", ctype=b"application/json", length=None):
    n = len(body) if length is None else length
    return b"HTTP/1.1 " + status + b"\r\nContent-Type: " + ctype + b"\r\nContent-Length: " + \
        str(n).encode() + b"\r\n\r\n" + body


def test_a_truncated_response_is_a_model_error(served):
    narr = RemoteNarrator(served[1], BG)
    with fake_server(http(b"abc", length=100)) as url:
        narr.url = url
        with pytest.raises(ModelError, match=f"Bad response from the book2audio server at {url}: IncompleteRead"):
            narr.generate("a", 1)


def test_a_200_that_is_not_audio_is_a_model_error(served):
    narr = RemoteNarrator(served[1], BG)
    with fake_server(http(b"not a wav file", ctype=b"audio/wav")) as url:
        narr.url = url
        with pytest.raises(ModelError, match=f"{url} did not return audio"):
            narr.generate("a", 1)


@pytest.mark.parametrize("info", [b'{"models": ["a"]}', b"[]", b'{"models": {"m": 5}}', b'{"models": {"m": {}}}',
                                  b"<html>", b'{"version": "1"}'])
def test_a_malformed_info_is_not_a_book2audio_server(info):
    with fake_server(http(info)) as url:
        with pytest.raises(ModelError, match=f"^{url} does not look like a book2audio server.$"):
            RemoteNarrator(url, "m")


def test_a_foreign_server_error_names_the_url():
    with fake_server(http(b"", status=b"404 File not found", ctype=b"text/html")) as url:
        with pytest.raises(ModelError, match=f"^HTTP 404 File not found from {url}/info$"):
            RemoteNarrator(url, BG)


def test_a_silent_server_is_a_timeout_not_unreachable(monkeypatch):
    monkeypatch.setattr(remote, "TIMEOUT", 0.3)
    with fake_server() as url:
        with pytest.raises(ModelError, match=f"^Timed out waiting for the book2audio server at {url} "):
            RemoteNarrator(url, BG)


def test_every_request_has_the_timeout(served, monkeypatch):
    seen, urlopen = [], urllib.request.urlopen
    monkeypatch.setattr(remote.urllib.request, "urlopen",
                        lambda req, timeout=None: seen.append(timeout) or urlopen(req, timeout=timeout))
    RemoteNarrator(served[1], BG).generate("a", 1)
    assert seen == [remote.TIMEOUT] * 2 and remote.TIMEOUT == 600


def test_uploads_are_remembered_across_narrators(served, tmp_path):
    ref = str(ref_file(tmp_path))
    RemoteNarrator(served[1], BG).generate("a", 1, ref_audio=ref)
    [f] = Path(served[0].refs.name).iterdir()
    mtime = f.stat().st_mtime_ns
    RemoteNarrator(served[1], DESIGN).generate("a", 1, ref_audio=ref)       # a new narrator, same process
    assert f.stat().st_mtime_ns == mtime
