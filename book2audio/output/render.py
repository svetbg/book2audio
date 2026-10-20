import hashlib
import shutil
import time

import numpy as np
import soundfile as sf

from book2audio.config import CHAPTER_LEAD, CHAPTER_TAIL, CHARS_PER_SEC
from book2audio.logging import log
from book2audio.output.audio import fmt_time, normalize, silence
from book2audio.tts.narrator import synth_chunk


def render_preview(narr, ch, plan, voice, args, workdir):
    sr = narr.sr
    parts, n = [], 0
    for kind, val in plan:
        if kind == "pause":
            parts.append(silence(val, sr))
            continue
        if n >= args.preview:
            break
        n += 1
        t0 = time.time()
        a = normalize(synth_chunk(narr, val, voice, args.seed, args.retries))
        el = time.time() - t0
        log(f"  preview {n}/{args.preview}: {len(a) / sr:4.1f}s audio in {el:5.1f}s "
              f"(x{el / max(len(a) / sr, 1e-3):.1f} real-time)", flush=True)
        parts.append(a)
    out = workdir / "preview.wav"
    sf.write(out, np.concatenate(parts) if parts else silence(1, sr), sr, subtype="PCM_16")
    log(f"\nPreview written to {out}")


def render_chapter(narr, idx, total, ch, plan, voice, voice_key, args, workdir):
    final = workdir / f"ch{idx:03d}.wav"
    if final.exists():
        log(f"[{idx}/{total}] {ch.title} — already done, skipping", flush=True)
        return final
    sr = narr.sr
    cdir = workdir / f"ch{idx:03d}_chunks"
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
        h = hashlib.md5((voice_key + "|" + val).encode("utf-8")).hexdigest()[:10]
        f = cdir / f"{h}.wav"                      # keyed by text, survives reordering
        if (cdir / f"{k:04d}_{h}.wav").exists():   # chunks from older script versions
            f = cdir / f"{k:04d}_{h}.wav"
        if f.exists():
            a, _ = sf.read(f, dtype="float32")
        else:
            t0 = time.time()
            a = normalize(synth_chunk(narr, val, voice, args.seed, args.retries))
            tmp = f.with_name(f.stem + ".tmp.wav")      # atomic: no half-written chunks
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

    tmp = workdir / f"ch{idx:03d}.tmp.wav"
    sf.write(tmp, np.concatenate(parts), sr, subtype="PCM_16")
    tmp.rename(final)
    if not args.keep_chunks:
        shutil.rmtree(cdir, ignore_errors=True)
    return final
