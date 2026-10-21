import glob
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
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
    main(monkeypatch, "--host", "localhost", "--port", str(port))
    [(address, narrators)] = started
    assert address == ("localhost", port) and list(narrators) == [BG, DESIGN]
    assert f"book2audio server on http://localhost:{port} ({BG}, {DESIGN})" in capsys.readouterr().out


def test_main_serves_only_the_models_asked_for(monkeypatch, started):
    main(monkeypatch, "--port", "0", "--model", "a/b", "--model", "a/b", "--model", "c/d")
    assert list(started[0][1]) == ["a/b", "c/d"]


def test_main_defaults_to_port_8765_on_localhost(monkeypatch, started):
    class Stop(Exception):
        pass

    def refuse(address, narrators):
        started.append((address, narrators))
        raise Stop                      # don't bind the real 8765

    monkeypatch.setattr(server_mod, "Server", refuse)
    with pytest.raises(Stop):
        main(monkeypatch)
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


def test_help_shows_the_port_default_once_and_no_none_defaults(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["book2audio-server", "--help"])
    with pytest.raises(SystemExit):
        server_mod.parse_arguments()
    out = " ".join(capsys.readouterr().out.split())
    assert "port to listen on (default: 8765)" in out
    assert "(default: None)" not in out and out.count("(default: PyTorch") == 0
    assert "(default: 127.0.0.1)" in out


def test_handlers_have_a_read_timeout():
    assert 0 < server_mod.Handler.timeout <= 600


@pytest.fixture(autouse=True)
def keep_sigterm_handler():
    old = signal.getsignal(signal.SIGTERM)      # main() installs its own
    yield
    signal.signal(signal.SIGTERM, old)


def test_a_peer_trickling_bytes_is_cut_off_at_the_deadline(monkeypatch, calls):
    monkeypatch.setattr(server_mod.Handler, "deadline", 0.5)       # each read stays well inside the 60 s timeout
    stop = threading.Event()
    with serve(BG) as (server, url):
        with socket.create_connection(("127.0.0.1", server.server_port)) as slow:
            def trickle():
                try:
                    slow.sendall(b"GET /info HTTP/1.1\r\nX-Slow: ")
                    while not stop.wait(0.1):
                        slow.sendall(b"a")
                except OSError:
                    pass
            t = threading.Thread(target=trickle, daemon=True)
            t.start()
            start = time.monotonic()
            with urllib.request.urlopen(url + "/info", timeout=10) as r:
                assert r.status == 200
            assert time.monotonic() - start < 5
            stop.set()


def test_a_slow_but_complete_upload_is_not_cut_off_while_generating(monkeypatch, calls):
    monkeypatch.setattr(server_mod.Handler, "deadline", 0.3)
    with serve(BG) as (server, url):
        body = json.dumps({"model": BG, "text": "hi", "seed": 1}).encode()
        orig = server.narrators[BG].generate

        def slow(*a, **kw):
            time.sleep(0.8)                 # past the deadline: the body was already in
            return orig(*a, **kw)
        server.narrators[BG].generate = slow
        req = urllib.request.Request(url + "/generate", body, method="POST")
        with urllib.request.urlopen(req, timeout=10) as r:
            assert r.status == 200


@pytest.mark.parametrize("method", ["PUT", "POST"])
def test_a_huge_content_length_is_a_413(calls, method):
    path = b"/refs/" + b"0" * 64 if method == "PUT" else b"/generate"
    with serve(BG) as (server, url):
        for n in (server_mod.MAX_BODY + 1, 99999999999, 10 ** 24):
            status, _, body = raw(server, method.encode() + b" " + path + b" HTTP/1.1\r\nContent-Length: "
                                  + str(n).encode() + b"\r\n\r\n")
            assert status == 413 and "error" in body
        assert urllib.request.urlopen(url + "/info").status == 200


def test_head_gets_the_error_headers_and_no_body(calls):
    with serve(BG) as (server, url):
        with socket.create_connection(("127.0.0.1", server.server_port), timeout=5) as c:
            c.sendall(b"HEAD /info HTTP/1.1\r\n\r\n")
            c.shutdown(socket.SHUT_WR)
            data = b""
            while chunk := c.recv(4096):
                data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    assert head.startswith(b"HTTP/1.0 501") and body == b""


def test_sigterm_stops_the_server_and_removes_the_refs_dir(tmp_path):
    code = ("import sys; sys.path.insert(0, 'tests'); from stubs import stub_modules; "
            "import pytest; mp = pytest.MonkeyPatch(); stub_modules(mp); "
            "from book2audio import server; sys.argv = ['book2audio-server', '--port', '0']; server.main()")
    env = {**os.environ, "TMPDIR": str(tmp_path)}
    p = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE, text=True, env=env)
    while "book2audio server on" not in p.stdout.readline():
        pass
    assert glob.glob(str(tmp_path / "book2audio-refs-*"))
    p.send_signal(signal.SIGTERM)
    assert p.wait(timeout=10) == 0
    assert not glob.glob(str(tmp_path / "book2audio-refs-*"))
