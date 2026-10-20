import shutil
import time

import numpy as np
import soundfile as sf

from book2audio.config import CHAPTER_LEAD, CHAPTER_TAIL, CHARS_PER_SEC
from book2audio.logging import log
from book2audio.output.audio import fmt_time, normalize, silence
from book2audio.tts.narrator import Narrator
from book2audio.workdir import WorkDir


class ChapterRenderer:
    """Narrates chapter plans into the work folder, reusing finished chunks and chapters."""

    def __init__(self, narrator: Narrator, voice: dict, voice_key: str, workdir: WorkDir,
                 seed: int, retries: int, keep_chunks: bool = False):
        self.narrator = narrator
        self.voice = voice
        self.voice_key = voice_key
        self.workdir = workdir
        self.seed = seed
        self.retries = retries
        self.keep_chunks = keep_chunks

    def _synth(self, text):
        return normalize(self.narrator.synth_chunk(text, self.voice, self.seed, self.retries))

    def preview(self, plan, count):
        sr = self.narrator.sr
        parts, n = [], 0
        for kind, val in plan:
            if kind == "pause":
                parts.append(silence(val, sr))
                continue
            if n >= count:
                break
            n += 1
            t0 = time.time()
            a = self._synth(val)
            el = time.time() - t0
            log(f"  preview {n}/{count}: {len(a) / sr:4.1f}s audio in {el:5.1f}s "
                  f"(x{el / max(len(a) / sr, 1e-3):.1f} real-time)", flush=True)
            parts.append(a)
        out = self.workdir.preview_wav
        sf.write(out, np.concatenate(parts) if parts else silence(1, sr), sr, subtype="PCM_16")
        log(f"\nPreview written to {out}")

    def render(self, idx, total, ch, plan):
        wd = self.workdir
        final = wd.chapter_wav(idx)
        if final.exists():
            log(f"[{idx}/{total}] {ch.title} — already done, skipping", flush=True)
            return final
        sr = self.narrator.sr
        cdir = wd.chunk_dir(idx)
        cdir.mkdir(exist_ok=True)
        tts_total = sum(1 for k, _ in plan if k == "tts")
        chars_total = sum(len(v) for k, v in plan if k == "tts")
        log(f"[{idx}/{total}] {ch.title}  ({tts_total} chunks, ~{fmt_time(chars_total / CHARS_PER_SEC)} audio)",
              flush=True)

        parts = [silence(CHAPTER_LEAD, sr)]
        k = done_chars = synth_chars = 0
        synth_secs = 0.0
        for kind, val in plan:
            if kind == "pause":
                parts.append(silence(val, sr))
                continue
            k += 1
            f = wd.chunk_file(idx, k, self.voice_key, val)
            if f.exists():
                a, _ = sf.read(f, dtype="float32")
            else:
                t0 = time.time()
                a = self._synth(val)
                tmp = wd.chunk_tmp(f)      # atomic: no half-written chunks
                sf.write(tmp, a, sr, subtype="PCM_16")
                tmp.rename(f)
                el = time.time() - t0
                synth_secs += el
                synth_chars += len(val)
                eta = (chars_total - done_chars - len(val)) * synth_secs / max(synth_chars, 1)
                log(f"  {k:4d}/{tts_total}  {len(a) / sr:5.1f}s audio in {el:5.1f}s "
                      f"(x{el / max(len(a) / sr, 1e-3):.1f} real-time)  chapter ETA {fmt_time(eta)}",
                      flush=True)
            done_chars += len(val)
            parts.append(a)
        parts.append(silence(CHAPTER_TAIL, sr))

        tmp = wd.chapter_tmp(idx)
        sf.write(tmp, np.concatenate(parts), sr, subtype="PCM_16")
        tmp.rename(final)
        if not self.keep_chunks:
            shutil.rmtree(cdir, ignore_errors=True)
        return final
