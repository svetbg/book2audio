import argparse
import hashlib
import io
import json
import re
import signal
import socket
import sys
import tempfile
import threading
import traceback
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import soundfile as sf

from book2audio import __version__
from book2audio.config import DEFAULT_DESIGN_MODEL, DEFAULT_MODEL, DEFAULT_PORT
from book2audio.exceptions import Book2AudioException
from book2audio.logging import error, log
from book2audio.tts.narrator import Narrator, set_miopen_env

SHA256 = re.compile(r"[0-9a-f]{64}")
MAX_BODY = 200 * 2**20      # bytes; far above any reference WAV, below what could exhaust memory


class RequestError(Exception):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status


class Handler(BaseHTTPRequestHandler):
    """One request at a time (plain HTTPServer), so seeds and gen_kwargs never interleave."""

    timeout = 60    # seconds a socket read may stall; a dead client must not wedge the only thread
    deadline = 120  # seconds to receive the whole request; a peer trickling bytes must not wedge it either

    def setup(self):
        super().setup()
        self._deadline = threading.Timer(self.deadline, self._cut)
        self._deadline.daemon = True
        self._deadline.start()

    def _cut(self):
        try:
            self.connection.shutdown(socket.SHUT_RDWR)      # wakes the blocked read with EOF
        except OSError:
            pass

    def finish(self):
        self._deadline.cancel()
        super().finish()

    def handle_one_request(self):
        try:
            super().handle_one_request()
        except OSError:                 # cut off by the deadline, or the client left
            self.close_connection = True

    def _send(self, status, body=b"", ctype="application/json"):
        self.send_response(status)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if getattr(self, "command", None) != "HEAD" and status not in (204, 304):
            self.wfile.write(body)

    def _error(self, status, msg):
        self._send(status, json.dumps({"error": msg}).encode())

    def _body(self):
        try:
            n = int(self.headers.get("Content-Length") or 0)
        except ValueError as e:
            raise RequestError(400, f"bad request: {e}")
        if n < 0:
            raise RequestError(400, "bad request: negative Content-Length")
        if n > MAX_BODY:
            raise RequestError(413, f"request body over {MAX_BODY} bytes")
        data = self.rfile.read(n)
        self._deadline.cancel()         # received in time; generating may take longer
        if len(data) < n:
            raise RequestError(400, "bad request: incomplete body")
        return data

    def send_error(self, code, message=None, explain=None):
        """Unsupported methods, bad request lines: JSON like every other error, not an HTML page."""
        self._error(code, message or self.responses[code][0])

    def do_GET(self):
        if self.path != "/info":
            return self._error(404, f"unknown path {self.path}")
        models = {name: {"sample_rate": n.sr} for name, n in self.server.narrators.items()}
        self._send(200, json.dumps({"version": __version__, "models": models}).encode())

    def do_PUT(self):
        sha = self.path.removeprefix("/refs/")
        if sha == self.path or not SHA256.fullmatch(sha):
            return self._error(404, f"unknown path {self.path}")
        try:
            data = self._body()
        except RequestError as e:
            return self._error(e.status, str(e))
        if hashlib.sha256(data).hexdigest() != sha:
            return self._error(400, "the SHA-256 of the body does not match the path")
        self.server.ref_path(sha).write_bytes(data)
        self._send(204)

    def do_POST(self):
        if self.path != "/generate":
            return self._error(404, f"unknown path {self.path}")
        try:
            body = json.loads(self._body())
            model, text, seed = body["model"], body["text"], body["seed"]
        except RequestError as e:
            return self._error(e.status, str(e))
        except (ValueError, KeyError, TypeError) as e:
            return self._error(400, f"bad request: {type(e).__name__}: {e}")
        narr = self.server.narrators.get(model)
        if narr is None:
            return self._error(400, f"unknown model '{model}'. "
                                    f"This server has: {', '.join(self.server.narrators)}")
        try:
            ref = None
            if body.get("ref"):
                ref = self.server.ref_path(body["ref"]) if SHA256.fullmatch(body["ref"]) else None
                if ref is None or not ref.exists():
                    return self._error(404, "unknown ref")
            narr.gen_kwargs = body.get("gen_kwargs") or {}
            a = narr.generate(text, seed, ref_audio=str(ref) if ref else None,
                              ref_text=body.get("ref_text"), instruct=body.get("instruct"))
            buf = io.BytesIO()
            sf.write(buf, a, narr.sr, format="WAV", subtype="FLOAT")
            self._send(200, buf.getvalue(), "audio/wav")
        except Book2AudioException as e:
            self._error(400, str(e))
        except Exception as e:
            traceback.print_exc()
            self._error(500, f"{type(e).__name__}: {e}")


class Server(HTTPServer):
    """narrators: {model name: Narrator}, the allowlist. Uploaded references live as long as it does."""

    def __init__(self, address, narrators):
        self.narrators = narrators
        self.refs = tempfile.TemporaryDirectory(prefix="book2audio-refs-")   # before the bind: server_close needs it
        super().__init__(address, Handler)

    def ref_path(self, sha):
        return Path(self.refs.name) / f"{sha}.wav"

    def server_close(self):
        super().server_close()
        self.refs.cleanup()


class Formatter(argparse.ArgumentDefaultsHelpFormatter):
    def _get_help_string(self, action):
        return action.help if action.default is None else super()._get_help_string(action)


def parse_arguments():
    ap = argparse.ArgumentParser(
        prog="book2audio-server",
        description="Run OmniVoice for book2audio clients (book2audio --server URL). "
                    "No authentication: bind to a trusted network or use an SSH tunnel.",
        formatter_class=Formatter)
    ap.add_argument("--host", default="127.0.0.1", help="address to listen on")
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="port to listen on")
    ap.add_argument("--model", action="append", metavar="M",
                    help=f"model to serve, repeatable; the only ones clients may use "
                         f"(without it: {DEFAULT_MODEL} and {DEFAULT_DESIGN_MODEL})")
    ap.add_argument("--device", default="cpu", help="cpu, or cuda once ROCm PyTorch is installed")
    ap.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                    help="auto = float32 on CPU, float16 on GPU (try float32 if GPU audio is garbled)")
    ap.add_argument("--no-miopen", action="store_true",
                    help="ROCm: don't use MIOpen (try this if the GPU crashes or hangs)")
    ap.add_argument("--threads", type=int, help="CPU threads for PyTorch (PyTorch decides if not given)")
    return ap.parse_args()


def main():
    args = parse_arguments()
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))    # how a service is stopped: still run the cleanup below
    narrators, server = {}, None
    try:
        server = Server((args.host, args.port), narrators)   # bind first: a busy port fails before any model loads
        set_miopen_env(args.device)          # before torch initialises, so before any Narrator
        # ponytail: all models stay loaded at once, as in local voice design. If VRAM runs
        # short, load the design model on demand instead.
        for m in dict.fromkeys(args.model or [DEFAULT_MODEL, DEFAULT_DESIGN_MODEL]):
            narrators[m] = Narrator(m, args.device, args.threads, {}, args.dtype, args.no_miopen)
    except (Book2AudioException, OSError) as e:
        if server:
            server.server_close()
        error(f"\nERROR: {e}")
        sys.exit(1)
    log(f"book2audio server on http://{args.host}:{server.server_port} "
        f"({', '.join(narrators)}). Ctrl+C stops it.", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
