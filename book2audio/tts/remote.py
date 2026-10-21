import hashlib
import http.client
import io
import json
import urllib.error
import urllib.request
from pathlib import Path

import soundfile as sf

from book2audio.exceptions import ModelError
from book2audio.logging import log
from book2audio.tts.narrator import Narrator

TIMEOUT = 600   # seconds per request; models are loaded at server start, so one generation is the longest wait


class ServerError(ModelError):
    def __init__(self, status, msg):
        super().__init__(msg)
        self.status = status


class RemoteNarrator(Narrator):
    """A Narrator whose generate() runs on a book2audio-server. Needs neither torch nor omnivoice,
    so it skips Narrator.__init__ and inherits synth_chunk unchanged."""

    _uploaded = set()       # (server url, sha256) of references sent, once per process

    def __init__(self, url, model_name, gen_kwargs=None):
        self.url, self.model_name = url.rstrip("/"), model_name
        self.gen_kwargs = gen_kwargs or {}
        try:
            models = {n: m["sample_rate"] for n, m in json.loads(self._call("GET", "/info"))["models"].items()}
        except (ValueError, KeyError, TypeError, AttributeError):
            raise ModelError(f"{self.url} does not look like a book2audio server.")
        if model_name not in models:
            raise ModelError(f"The book2audio server at {self.url} has no model '{model_name}'. "
                             f"It has: {', '.join(models)}")
        self.sr = models[model_name]
        log(f"Using book2audio server at {self.url} ({model_name})", flush=True)

    def with_model(self, model_name):
        return RemoteNarrator(self.url, model_name, {})

    def generate(self, text, seed, ref_audio=None, ref_text=None, instruct=None):
        sha = self._upload(ref_audio) if ref_audio else None
        body = dict(model=self.model_name, text=text, seed=seed, ref=sha, ref_text=ref_text,
                    instruct=instruct, gen_kwargs=self.gen_kwargs)
        try:
            wav = self._call("POST", "/generate", json.dumps(body).encode(), "application/json")
        except ServerError as e:
            if e.status != 404 or not sha:
                raise
            self._uploaded.discard((self.url, sha))  # the server forgot it (restarted): send it again
            body["ref"] = self._upload(ref_audio)
            wav = self._call("POST", "/generate", json.dumps(body).encode(), "application/json")
        try:
            return sf.read(io.BytesIO(wav), dtype="float32")[0]
        except sf.SoundFileError as e:
            raise ModelError(f"The book2audio server at {self.url} did not return audio: {e}")

    def _upload(self, path):
        data = Path(path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if (self.url, sha) not in self._uploaded:
            self._call("PUT", f"/refs/{sha}", data, "audio/wav")
            self._uploaded.add((self.url, sha))
        return sha

    def _call(self, method, path, data=None, ctype=None):
        req = urllib.request.Request(self.url + path, data, {"Content-Type": ctype} if ctype else {},
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read())["error"]
            except (ValueError, KeyError, TypeError):
                msg = f"HTTP {e.code} {e.reason} from {req.full_url}"
            raise ServerError(e.code, msg)
        except http.client.HTTPException as e:
            raise ModelError(f"Bad response from the book2audio server at {self.url}: {type(e).__name__}: {e}")
        except (urllib.error.URLError, OSError) as e:
            if type(e) is TimeoutError:             # connected, then no answer
                raise ModelError(f"Timed out waiting for the book2audio server at {self.url} ({TIMEOUT} s). "
                                 f"If it was just started, it may still be loading models.")
            raise ModelError(f"Cannot reach the book2audio server at {self.url}: "
                             f"{getattr(e, 'reason', e)}")
