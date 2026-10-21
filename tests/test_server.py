import glob
import json
import os
import socket
import sys
import tempfile
import urllib.request

import pytest

from book2audio import server as server_mod
from stubs import serve, stub_modules

BG, DESIGN = "preslaff/omnivoice-bg", "k2-fsa/OmniVoice"


@pytest.fixture
def calls(monkeypatch):
    return stub_modules(monkeypatch)


@pytest.fixture
def started(monkeypatch, calls):
    """main() without serving: records (address, narrators) of each Server it builds."""
    seen = []

    class Recording(server_mod.Server):
        def __init__(self, address, narrators):
            seen.append((address, narrators))
            super().__init__(address, narrators)

        def serve_forever(self, *a):
            pass
    monkeypatch.setattr(server_mod, "Server", Recording)
    return seen


def main(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["book2audio-server", *argv])
    server_mod.main()


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_main_serves_the_default_models_on_the_given_host_and_port(monkeypatch, started, capsys):
    port = free_port()
    main(monkeypatch, "--host", "127.0.0.1", "--port", str(port))
    [(address, narrators)] = started
    assert address == ("127.0.0.1", port) and list(narrators) == [BG, DESIGN]
    assert f"book2audio server on http://127.0.0.1:{port} ({BG}, {DESIGN})" in capsys.readouterr().out


def test_main_serves_only_the_models_asked_for(monkeypatch, started):
    main(monkeypatch, "--port", "0", "--model", "a/b", "--model", "a/b", "--model", "c/d")
    assert list(started[0][1]) == ["a/b", "c/d"]


def test_main_defaults_to_port_8765_on_localhost(monkeypatch, started):
    try:
        main(monkeypatch)
    except OSError:
        pytest.skip("port 8765 is in use")
    assert started[0][0] == ("127.0.0.1", 8765)


def test_main_sets_the_miopen_env_before_any_model_loads(monkeypatch, started):
    calls = stub_modules(monkeypatch, gpu=True)
    for k in ("MIOPEN_LOG_LEVEL", "MIOPEN_FIND_MODE"):
        monkeypatch.delenv(k, raising=False)
    model, envs = sys.modules["omnivoice"].OmniVoice, []
    load = model.from_pretrained.__func__

    def spy(cls, name, **kw):
        envs.append((os.environ.get("MIOPEN_FIND_MODE"), os.environ.get("MIOPEN_LOG_LEVEL")))
        return load(cls, name, **kw)
    monkeypatch.setattr(model, "from_pretrained", classmethod(spy))
    main(monkeypatch, "--port", "0", "--device", "cuda")
    assert envs == [("FAST", "3")] * 2 and len(calls.loads) == 2


def refs_dirs():
    return set(glob.glob(os.path.join(tempfile.gettempdir(), "book2audio-refs-*")))


def test_a_busy_port_fails_before_any_model_loads(monkeypatch, calls, capsys):
    before = refs_dirs()
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        s.listen()
        with pytest.raises(SystemExit) as e:
            main(monkeypatch, "--port", str(s.getsockname()[1]))
    assert e.value.code == 1 and calls.loads == []
    err = capsys.readouterr().err
    assert err.startswith("\nERROR: ") and "Traceback" not in err
    assert refs_dirs() == before


def test_a_model_that_fails_to_load_frees_the_port(monkeypatch, calls, capsys):
    stub_modules(monkeypatch, gpu=False)
    port = free_port()
    with pytest.raises(SystemExit) as e:
        main(monkeypatch, "--port", str(port), "--device", "cuda")      # no GPU on the stub
    assert e.value.code == 1 and "sees no GPU" in capsys.readouterr().err
    with socket.socket() as s:
        s.bind(("127.0.0.1", port))


def test_a_stalled_request_does_not_block_the_next_one(monkeypatch, calls):
    monkeypatch.setattr(server_mod.Handler, "timeout", 0.3)
    with serve(BG) as (server, url):
        with socket.create_connection(("127.0.0.1", server.server_port)) as stalled:
            stalled.sendall(b"PUT /refs/" + b"0" * 64 + b" HTTP/1.1\r\nContent-Length: 100\r\n\r\nabc")
            with urllib.request.urlopen(url + "/info", timeout=5) as r:       # waits ~0.3 s, not forever
                assert r.status == 200


def raw(server, request):
    with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as c:
        c.sendall(request)
        c.shutdown(socket.SHUT_WR)
        data = b""
        while chunk := c.recv(4096):
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    return int(head.split()[1]), head, json.loads(body)


@pytest.mark.parametrize("length", [b"abc", b"-1"])
def test_put_with_an_invalid_content_length_is_a_400(calls, length):
    with serve(BG) as (server, url):
        status, _, body = raw(server, b"PUT /refs/" + b"0" * 64 + b" HTTP/1.1\r\nContent-Length: " + length + b"\r\n\r\n")
        assert status == 400 and body["error"].startswith("bad request: ")
        assert urllib.request.urlopen(url + "/info").status == 200


def test_unsupported_methods_get_a_json_error(calls):
    with serve(BG) as (server, url):
        status, head, body = raw(server, b"DELETE /info HTTP/1.1\r\n\r\n")
        assert status == 501 and b"application/json" in head and "error" in body
