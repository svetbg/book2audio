import hashlib
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

    def __init__(self, url, model_name, gen_kwargs=None):
        self.url, self.model_name = url.rstrip("/"), model_name
        self.gen_kwargs = gen_kwargs or {}
        self._uploaded = set()
        try:
            models = json.loads(self._call("GET", "/info"))["models"]
        except (ValueError, KeyError, TypeError):
            raise ModelError(f"{self.url} does not look like a book2audio server.")
        if model_name not in models:
            raise ModelError(f"The book2audio server at {self.url} has no model '{model_name}'. "
                             f"It has: {', '.join(models)}")
        self.sr = models[model_name]["sample_rate"]
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
            self._uploaded.discard(sha)             # the server forgot it (restarted): send it again
            body["ref"] = self._upload(ref_audio)
            wav = self._call("POST", "/generate", json.dumps(body).encode(), "application/json")
        return sf.read(io.BytesIO(wav), dtype="float32")[0]

    def _upload(self, path):
        data = Path(path).read_bytes()
        sha = hashlib.sha256(data).hexdigest()
        if sha not in self._uploaded:
            self._call("PUT", f"/refs/{sha}", data, "audio/wav")
            self._uploaded.add(sha)
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
                msg = f"HTTP {e.code} {e.reason}"
            raise ServerError(e.code, msg)
        except (urllib.error.URLError, OSError) as e:
            raise ModelError(f"Cannot reach the book2audio server at {self.url}: "
                             f"{getattr(e, 'reason', e)}")
