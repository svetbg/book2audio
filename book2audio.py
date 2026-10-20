#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
book2audio.py - Turn an EPUB or TXT ebook into a chaptered M4B audiobook
using OmniVoice (default model: preslaff/omnivoice-bg, a Bulgarian fine-tune).

Runs on the CPU by default. Later, with a ROCm build of PyTorch, add --device cuda.

Two voice modes
  1) With a sample:    --ref narrator.wav [--ref-text "exact words spoken in the sample"]
  2) Without a sample: [--voice "female, low pitch"]
     A voice is designed once, saved as designed_voice.wav in the work folder,
     and then used as the reference for the whole book, so it stays consistent.

Typical workflow
  python book2audio.py book.epub --list                      # check chapter detection
  python book2audio.py book.epub --ref narrator.wav --preview  # ~6 chunks -> preview.wav
  python book2audio.py book.epub --ref narrator.wav --chapters 1-2   # test run
  python book2audio.py book.epub --ref narrator.wav          # whole book -> book.m4b

  python book2audio.py book.epub --voice "male, low pitch" --design-only   # audition voice
  (the voice is designed with the base model k2-fsa/OmniVoice, see --design-model)
  python book2audio.py book.epub --voice "male, low pitch"

Re-running the same command resumes where it stopped (finished chunks and
chapters are cached in the work folder).
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

DEFAULT_MODEL = "preslaff/omnivoice-bg"
DEFAULT_SR = 24000

# Pauses (seconds)
PAUSE_CHUNK = 0.30       # between chunks inside one paragraph
PAUSE_PARAGRAPH = 0.70   # between paragraphs
PAUSE_SCENE = 1.40       # scene breaks such as "* * *"
CHAPTER_LEAD = 0.8
CHAPTER_TAIL = 2.0

# Spoken once to create a voice when no sample is given; reused as ref_text.
DESIGN_SENTENCE = ("Добре дошли. С този глас ще ви прочета цялата книга – "
                   "спокойно, ясно и глава по глава.")

CHARS_PER_SEC = 13.0     # rough narration speed, used for sanity checks / estimates


# --------------------------------------------------------------------------- #
# Book model
# --------------------------------------------------------------------------- #
@dataclass
class Chapter:
    title: str
    paragraphs: list = field(default_factory=list)

    @property
    def chars(self) -> int:
        return sum(len(p) for p in self.paragraphs)


@dataclass
class Book:
    title: str
    author: str
    chapters: list
    cover: Optional[bytes] = None
    cover_ext: str = "jpg"


def die(msg: str, code: int = 1):
    print(f"\nERROR: {msg}", file=sys.stderr)
    sys.exit(code)


def clean_text(s: str) -> str:
    s = s.replace("\u00ad", "")            # soft hyphens
    s = s.replace("\u00a0", " ")           # non-breaking spaces
    s = re.sub(r"\s+", " ", s)             # keeps U+0301 stress marks intact
    return s.strip()


def has_speech(s: str) -> bool:
    return re.search(r"[^\W_]", s) is not None


def fill_missing_titles(chapters):
    for i, ch in enumerate(chapters, 1):
        if not ch.title:
            first = ch.paragraphs[0] if ch.paragraphs else ""
            ch.title = first if 0 < len(first) <= 80 else f"Глава {i}"


# --------------------------------------------------------------------------- #
# Loaders
# --------------------------------------------------------------------------- #
BLOCK_TAGS = ["p", "h1", "h2", "h3", "h4", "h5", "h6", "li", "blockquote",
              "div", "pre", "td", "dt", "dd"]


def _flatten_toc(toc):
    out = []
    for entry in toc:
        if isinstance(entry, (list, tuple)):
            sec = entry[0]
            children = entry[1] if len(entry) > 1 else []
            if getattr(sec, "href", None):
                out.append((sec.href, sec.title))
            out.extend(_flatten_toc(children))
        elif getattr(entry, "href", None):
            out.append((entry.href, entry.title))
    return out


def load_epub(path: Path, min_chars: int) -> Book:
    import warnings
    warnings.filterwarnings("ignore")
    try:
        import ebooklib
        from ebooklib import epub
        from bs4 import BeautifulSoup
    except ImportError:
        die("Missing packages. Run: pip install ebooklib beautifulsoup4")

    book = epub.read_epub(str(path))

    def meta(name):
        try:
            v = book.get_metadata("DC", name)
            return clean_text(v[0][0]) if v else ""
        except Exception:
            return ""

    # Chapter titles from the table of contents (matched by file name)
    toc_titles = {}
    try:
        for href, title in _flatten_toc(book.toc):
            key = Path(href.split("#")[0]).name
            if key and title and key not in toc_titles:
                toc_titles[key] = clean_text(title)
    except Exception:
        pass

    chapters = []
    for idref, _linear in book.spine:
        item = book.get_item_with_id(idref)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue
        soup = BeautifulSoup(item.get_content(), "html.parser")
        for t in soup(["script", "style", "head"]):
            t.decompose()
        for br in soup.find_all("br"):
            br.replace_with("\n")
        body = soup.body or soup

        paragraphs = []
        for el in body.find_all(BLOCK_TAGS):
            if el.find(BLOCK_TAGS):        # only leaf blocks, avoids duplicates
                continue
            txt = clean_text(el.get_text())
            if txt:
                paragraphs.append(txt)
        if not paragraphs:
            txt = clean_text(body.get_text())
            if txt:
                paragraphs = [txt]

        ch = Chapter(title="", paragraphs=paragraphs)
        if ch.chars < min_chars:           # cover, copyright page, TOC, ...
            continue
        heading = body.find(["h1", "h2", "h3"])
        ch.title = (toc_titles.get(Path(item.get_name()).name)
                    or (clean_text(heading.get_text()) if heading else ""))
        chapters.append(ch)

    fill_missing_titles(chapters)

    cover, ext = None, "jpg"
    try:
        items = list(book.get_items_of_type(ebooklib.ITEM_COVER))
        if not items:
            cid = book.get_metadata("OPF", "cover")
            if cid:
                it = book.get_item_with_id(cid[0][1].get("content"))
                if it is not None:
                    items = [it]
        if not items:
            items = [i for i in book.get_items_of_type(ebooklib.ITEM_IMAGE)
                     if "cover" in i.get_name().lower()]
        if items:
            cover = items[0].get_content()
            ext = "png" if items[0].get_name().lower().endswith(".png") else "jpg"
    except Exception:
        pass

    return Book(title=meta("title") or path.stem, author=meta("creator"),
                chapters=chapters, cover=cover, cover_ext=ext)


CHAPTER_LINE = re.compile(r"^\s*(глава|част|пролог|епилог|chapter|prologue|epilogue)\b",
                          re.IGNORECASE)


def read_text_file(path: Path) -> str:
    raw = path.read_bytes()
    for enc in ("utf-8-sig", "cp1251"):     # many Bulgarian .txt files are cp1251
        try:
            return raw.decode(enc)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def load_txt(path: Path, min_chars: int) -> Book:
    text = read_text_file(path).replace("\r\n", "\n")
    blocks = re.split(r"\n\s*\n", text)
    if len(blocks) < 3:                     # no blank lines: one paragraph per line
        blocks = text.split("\n")
    chapters, cur = [], Chapter(title="")
    for b in blocks:
        b = clean_text(b)
        if not b:
            continue
        if len(b) <= 80 and CHAPTER_LINE.match(b):
            if cur.paragraphs:
                chapters.append(cur)
            cur = Chapter(title=b, paragraphs=[b])
        else:
            cur.paragraphs.append(b)
    if cur.paragraphs:
        chapters.append(cur)
    chapters = [c for c in chapters if c.chars >= min_chars] or chapters
    fill_missing_titles(chapters)
    return Book(title=path.stem, author="", chapters=chapters)


def load_book(path: Path, min_chars: int) -> Book:
    suffix = path.suffix.lower()
    if suffix == ".epub":
        return load_epub(path, min_chars)
    if suffix in (".txt", ".text"):
        return load_txt(path, min_chars)
    die(f"Unsupported format '{suffix}'. Convert it to EPUB first, e.g. with Calibre:\n"
        f"  ebook-convert \"{path}\" \"{path.with_suffix('.epub')}\"")


# --------------------------------------------------------------------------- #
# Pronunciation lexicon (fix stress / readings)
# --------------------------------------------------------------------------- #
def load_lexicon(path: Optional[str]):
    """Lines: 'word<TAB>replacement' or 'word=replacement'. '#' starts a comment.
    Example (stress mark = U+0301 after the vowel):  бутона=буто́на"""
    rules = []
    if not path:
        return rules
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sep = "\t" if "\t" in line else ("=" if "=" in line else None)
        if not sep:
            continue
        src, dst = (x.strip() for x in line.split(sep, 1))
        if src:
            rules.append((re.compile(rf"(?<!\w){re.escape(src)}(?!\w)", re.IGNORECASE), dst))
    return rules


def apply_lexicon(text: str, rules) -> str:
    def repl(m, dst):
        if m.group(0)[:1].isupper() and dst:
            return dst[0].upper() + dst[1:]
        return dst
    for pat, dst in rules:
        text = pat.sub(lambda m, d=dst: repl(m, d), text)
    return text


# --------------------------------------------------------------------------- #
# Chunking
# --------------------------------------------------------------------------- #
SENTENCE_RE = re.compile(r"[^.!?…]+(?:[.!?…]+[\"»”“'’)\]]*|$)")


def pack(pieces, max_chars):
    out, cur = [], ""
    for p in pieces:
        if not cur:
            cur = p
        elif len(cur) + 1 + len(p) <= max_chars:
            cur += " " + p
        else:
            out.append(cur)
            cur = p
    if cur:
        out.append(cur)
    return out


def split_long(s, max_chars):
    if len(s) <= max_chars:
        return [s]
    for pat in (r"(?<=[,;:])\s+", r"\s+(?=[—–-]\s)", r"\s+"):
        parts = [x for x in re.split(pat, s) if x]
        if len(parts) > 1:
            out = []
            for part in pack(parts, max_chars):
                out.extend(split_long(part, max_chars) if len(part) > max_chars else [part])
            return out
    return [s[i:i + max_chars] for i in range(0, len(s), max_chars)]


def chunk_paragraph(para, max_chars):
    sentences = [s.strip() for s in SENTENCE_RE.findall(para) if s.strip()]
    pieces = []
    for s in sentences:
        pieces.extend(split_long(s, max_chars))
    return pack(pieces, max_chars)


def chapter_plan(ch: Chapter, max_chars: int, lexicon):
    plan = []
    for para in ch.paragraphs:
        para = apply_lexicon(para, lexicon)
        if not has_speech(para):
            plan.append(("pause", PAUSE_SCENE))
            continue
        chunks = [c for c in chunk_paragraph(para, max_chars) if has_speech(c)]
        for i, c in enumerate(chunks):
            plan.append(("tts", c))
            plan.append(("pause", PAUSE_CHUNK if i < len(chunks) - 1 else PAUSE_PARAGRAPH))
    return plan


# --------------------------------------------------------------------------- #
# Audio helpers
# --------------------------------------------------------------------------- #
def silence(secs, sr):
    return np.zeros(int(round(secs * sr)), dtype=np.float32)


def trim_silence(a, sr, rel_thresh=0.03, pad=0.06):
    if a.size == 0:
        return a
    peak = float(np.max(np.abs(a)))
    if peak <= 0:
        return a
    idx = np.where(np.abs(a) > rel_thresh * peak)[0]
    if idx.size == 0:
        return a
    s = max(0, idx[0] - int(pad * sr))
    e = min(len(a), idx[-1] + int(pad * sr))
    return a[s:e]


def normalize(a, target_dbfs=-21.0):
    if a.size == 0:
        return a
    rms = float(np.sqrt(np.mean(a ** 2)))
    if rms < 1e-6:
        return a
    a = a * (10 ** (target_dbfs / 20) / rms)
    peak = float(np.max(np.abs(a)))
    if peak > 0.97:
        a = a * (0.97 / peak)
    return a.astype(np.float32)


def expected_secs(text):
    return max(0.6, len(re.findall(r"[^\W_]", text)) / CHARS_PER_SEC)


def plausible(dur, text):
    e = expected_secs(text)
    return 0.4 * e - 0.3 <= dur <= 2.2 * e + 1.5


def fmt_time(secs):
    secs = int(max(0, secs))
    h, rem = divmod(secs, 3600)
    m, s = divmod(rem, 60)
    return f"{h}h{m:02d}m" if h else f"{m}m{s:02d}s"


# --------------------------------------------------------------------------- #
# OmniVoice wrapper
# --------------------------------------------------------------------------- #
class Narrator:
    def __init__(self, model_name, device="cpu", threads=None, gen_kwargs=None, dtype="auto",
                 no_miopen=False):
        try:
            import torch
        except ImportError:
            die("PyTorch is not installed (see the install steps at the top of the script).")
        self.torch = torch
        if threads:
            torch.set_num_threads(threads)
        try:
            from omnivoice import OmniVoice
        except ImportError:
            die("OmniVoice is not installed. Run: pip install omnivoice")

        if device != "cpu" and not torch.cuda.is_available():
            die(f"--device {device} requested, but PyTorch sees no GPU. Is the ROCm build of "
                f"PyTorch installed in this environment? Check with:\n"
                f"  python -c \"import torch; print(torch.version.hip, torch.cuda.is_available())\"")
        if no_miopen:
            torch.backends.cudnn.enabled = False   # on ROCm this switches off MIOpen
        if dtype == "auto":
            dtype = "float32" if device == "cpu" else "float16"
        dtype = getattr(torch, dtype)
        print(f"Loading {model_name} on {device} ({str(dtype).replace('torch.', '')}) ...",
              flush=True)
        model, last_err = None, None
        for kw in (dict(device_map=device, dtype=dtype), dict(device_map=device), {}):
            try:
                model = OmniVoice.from_pretrained(model_name, **kw)
                break
            except TypeError as e:
                last_err = e
            except OSError as e:
                die(f"Could not download/load '{model_name}'.\n"
                    f"The model is gated: open https://huggingface.co/{model_name}, accept the "
                    f"conditions, then run 'hf auth login' (or 'huggingface-cli login').\n{e}")
        if model is None:
            raise last_err
        if hasattr(model, "to"):
            try:
                model = model.to(device)
            except Exception:
                pass
        if hasattr(model, "eval"):
            model.eval()
        self.model = model
        self.sr = int(getattr(model, "sampling_rate", None)
                      or getattr(model, "sample_rate", None) or DEFAULT_SR)
        sig = inspect.signature(model.generate)
        self._params = set(sig.parameters)
        self._var_kw = any(p.kind == p.VAR_KEYWORD for p in sig.parameters.values())
        self.gen_kwargs = gen_kwargs or {}

    def accepts(self, name):
        return self._var_kw or name in self._params

    def generate(self, text, seed, ref_audio=None, ref_text=None, instruct=None):
        torch = self.torch
        torch.manual_seed(seed)
        np.random.seed(seed % (2 ** 32))
        kw = {"text": text}
        if ref_audio:
            kw["ref_audio"] = ref_audio
        if ref_text and self.accepts("ref_text"):
            kw["ref_text"] = ref_text
        if instruct:
            if not self.accepts("instruct"):
                die("This OmniVoice version has no 'instruct' (voice design) parameter. "
                    "Run without --voice, or use --ref with a sample.")
            kw["instruct"] = instruct
        kw.update(self.gen_kwargs)
        with torch.inference_mode():
            out = self.model.generate(**kw)
        if isinstance(out, (list, tuple)):
            out = out[0]
        if hasattr(out, "detach"):
            out = out.detach().float().cpu().numpy()
        a = np.asarray(out, dtype=np.float32).squeeze()
        if a.ndim > 1:
            a = a.mean(axis=0 if a.shape[0] < a.shape[-1] else 1)
        return a


def synth_chunk(narr, text, voice, seed, retries):
    """Generate one chunk; retry with another seed if the length looks broken."""
    best, best_err = None, None
    for attempt in range(retries + 1):
        a = trim_silence(narr.generate(text, seed + attempt * 1000, **voice), narr.sr)
        dur = len(a) / narr.sr
        if plausible(dur, text):
            return a
        err = abs(np.log(max(dur, 0.05) / expected_secs(text)))
        if best is None or err < best_err:
            best, best_err = a, err
    print(f"    ! suspicious length after {retries + 1} tries, keeping best: {text[:60]}…",
          flush=True)
    return best


# --------------------------------------------------------------------------- #
# Pipeline steps
# --------------------------------------------------------------------------- #
def need_ffmpeg():
    if not shutil.which("ffmpeg"):
        die("ffmpeg not found. Install it, e.g.: sudo apt install ffmpeg")


def run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def detect_pauses(path):
    """Start times (s) of pauses in an audio file, via ffmpeg silencedetect."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
                        "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]


def prepare_voice(narr, args, workdir: Path):
    """Returns (voice kwargs for generate, cache key)."""
    if args.ref:
        src = Path(args.ref).expanduser()
        if not src.exists():
            die(f"Reference file not found: {src}")
        full = workdir / "reference_full.wav"
        run_ffmpeg(["-i", str(src), "-ac", "1", "-ar", str(narr.sr), str(full)])
        dur = sf.info(str(full)).duration
        ref_text = (args.ref_text or "").strip()

        if dur > args.ref_max_secs + 0.05:
            pauses = [p for p in detect_pauses(full) if 3.0 < p < args.ref_max_secs]
            hint = (f"  Pauses (good cut points) at: {', '.join(f'{p:.1f}s' for p in pauses)}\n"
                    if pauses else "")
            cut = f"{pauses[-1] + 0.2:.1f}" if pauses else "15"
            if ref_text:
                die(f"The sample is {dur:.1f}s long, but only the first {args.ref_max_secs:g}s "
                    f"are used.\nThen --ref-text contains words that are NOT in the reference audio, "
                    f"and the model speaks those words inside the book.\n"
                    f"Cut the sample at a pause and give only the words spoken up to that point:\n"
                    f"{hint}  ffmpeg -i \"{src}\" -to {cut} -c copy sample_cut.wav\n"
                    f"Or use the whole sample with --ref-max-secs {int(dur) + 1} (slower on CPU).")
            print(f"Note: using only the first {args.ref_max_secs:g}s of the {dur:.1f}s sample.")

        ref = workdir / "reference.wav"
        run_ffmpeg(["-i", str(full), "-t", str(args.ref_max_secs), str(ref)])

        if ref_text:
            used = min(dur, args.ref_max_secs)
            rate = len(re.findall(r"[^\W_]", ref_text)) / max(used, 0.1)
            if re.search(r"\d", ref_text):
                print("Tip: write numbers in --ref-text as words, exactly as they are spoken "
                      "(e.g. 'хиляда осемстотин шейсет и шеста').")
            if not 8 <= rate <= 20:
                print(f"Warning: --ref-text has ~{rate:.0f} letters per second of audio "
                      f"(normal speech is ~12-16). It probably doesn't match the clip exactly, "
                      f"which can make words from the sample appear in the book.")
            if not re.search(r"[.!?…][\"»”“'’)]*$", ref_text):
                ref_text += "."
        else:
            print("Note: no --ref-text given. Giving the exact transcript of the sample "
                  "usually improves cloning.")
        key = f"ref:{src.resolve()}:{src.stat().st_mtime}:{ref_text}:{args.seed}:{args.ref_max_secs}"
        return {"ref_audio": str(ref), "ref_text": ref_text or None}, key

    designed = workdir / "designed_voice.wav"
    meta_f = workdir / "designed_voice.json"
    want = {"voice": args.voice, "seed": args.seed, "model": args.design_model or args.model}
    have = None
    if meta_f.exists():
        try:
            have = json.loads(meta_f.read_text(encoding="utf-8"))
        except Exception:
            pass
    if designed.exists() and have != want:
        print("Voice settings changed since designed_voice.wav was made: designing a new one.")
        designed.unlink()
    if not designed.exists():
        # The Bulgarian fine-tune may ignore voice descriptions, so by default the
        # voice is designed with the base OmniVoice model and only then cloned.
        dn = narr if want["model"] == args.model else Narrator(
            want["model"], args.device, args.threads, {}, args.dtype, args.no_miopen)
        print(f"No sample given: designing a voice ({args.voice or 'automatic'}) "
              f"with {want['model']} ...", flush=True)
        a = dn.generate(DESIGN_SENTENCE, args.seed, instruct=args.voice)
        sf.write(designed, normalize(trim_silence(a, dn.sr)), dn.sr, subtype="PCM_16")
        meta_f.write_text(json.dumps(want, ensure_ascii=False), encoding="utf-8")
        if dn is not narr:
            del dn
            import gc
            gc.collect()
        print(f"Saved {designed}\n  Listen to it. If you don't like it, rerun with "
              f"another --seed and/or --voice (it is regenerated automatically).")
    key = f"designed:{args.voice}:{args.seed}:{designed.stat().st_mtime}"
    return {"ref_audio": str(designed), "ref_text": DESIGN_SENTENCE}, key


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
        print(f"  preview {n}/{args.preview}: {len(a) / sr:4.1f}s audio in {el:5.1f}s "
              f"(x{el / max(len(a) / sr, 1e-3):.1f} real-time)", flush=True)
        parts.append(a)
    out = workdir / "preview.wav"
    sf.write(out, np.concatenate(parts) if parts else silence(1, sr), sr, subtype="PCM_16")
    print(f"\nPreview written to {out}")


def render_chapter(narr, idx, total, ch, plan, voice, voice_key, args, workdir):
    final = workdir / f"ch{idx:03d}.wav"
    if final.exists():
        print(f"[{idx}/{total}] {ch.title} — already done, skipping", flush=True)
        return final
    sr = narr.sr
    cdir = workdir / f"ch{idx:03d}_chunks"
    cdir.mkdir(exist_ok=True)
    tts_total = sum(1 for k, _ in plan if k == "tts")
    chars_total = sum(len(v) for k, v in plan if k == "tts")
    print(f"[{idx}/{total}] {ch.title}  ({tts_total} chunks, ~{fmt_time(chars_total / CHARS_PER_SEC)} audio)",
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
            print(f"  {k:4d}/{tts_total}  {len(a) / sr:5.1f}s audio in {el:5.1f}s "
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


def ffmeta_escape(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s or "")


def build_m4b(book: Book, files, titles, out: Path, workdir: Path, bitrate: str):
    lst = workdir / "concat.txt"
    with open(lst, "w", encoding="utf-8") as fh:
        for p in files:
            fh.write("file '" + str(Path(p).resolve()).replace("'", "'\\''") + "'\n")

    e = ffmeta_escape
    lines = [";FFMETADATA1", f"title={e(book.title)}", f"album={e(book.title)}",
             f"artist={e(book.author)}", f"album_artist={e(book.author)}", "genre=Audiobook"]
    t = 0
    for p, title in zip(files, titles):
        info = sf.info(str(p))
        dur = int(round(info.frames * 1000 / info.samplerate))
        lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={t}", f"END={t + dur}",
                  f"title={e(title)}"]
        t += dur
    meta = workdir / "chapters.ffmeta"
    meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

    cmd = ["-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(meta)]
    cover = None
    if book.cover:
        cover = workdir / f"cover.{book.cover_ext}"
        cover.write_bytes(book.cover)
        cmd += ["-i", str(cover)]
    cmd += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
    if cover:
        cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
    cmd += ["-c:a", "aac", "-b:a", bitrate, "-ac", "1", "-movflags", "+faststart",
            "-f", "mp4", str(out)]
    print(f"\nEncoding {out.name} ({fmt_time(t / 1000)}) ...", flush=True)
    try:
        run_ffmpeg(cmd)
    except subprocess.CalledProcessError:
        if not cover:
            raise
        print("Cover could not be embedded, retrying without it ...")
        cmd = [x for x in cmd]  # rebuild without cover input/mapping
        i = cmd.index(str(cover))
        del cmd[i - 1:i + 1]
        j = cmd.index("2:v")
        del cmd[j - 1:j + 5]
        run_ffmpeg(cmd)
    print(f"Done: {out}")


def compact_ranges(nums):
    """[1,2,3,5,7,8] -> '1-3, 5, 7-8'"""
    out, start, prev = [], None, None
    for n in nums + [None]:
        if start is None:
            start = prev = n
        elif n is not None and n == prev + 1:
            prev = n
        else:
            out.append(f"{start}-{prev}" if prev != start else f"{start}")
            start = prev = n
    return ", ".join(out)


def parse_chapters(spec, n):
    if not spec:
        return list(range(1, n + 1))
    sel = set()
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            sel.update(range(int(a) if a else 1, (int(b) if b else n) + 1))
        else:
            sel.add(int(part))
    return sorted(i for i in sel if 1 <= i <= n)


# --------------------------------------------------------------------------- #
# Remembered voices
# --------------------------------------------------------------------------- #
VOICE_FIELDS = ("ref", "ref_text", "voice", "seed", "design_model", "ref_max_secs")


def voices_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    return base / "book2audio" / "voices"


def use_named_voice(args):
    d = voices_dir() / args.narrator
    ref, txt = d / "reference.wav", d / "ref_text.txt"
    if not ref.exists():
        have = sorted(p.name for p in voices_dir().glob("*") if (p / "reference.wav").exists())
        die(f"No saved voice named '{args.narrator}'. "
            f"Saved voices: {', '.join(have) if have else 'none yet (use --save-voice NAME)'}")
    args.ref = str(ref)
    args.ref_text = txt.read_text(encoding="utf-8").strip() if txt.exists() else None
    print(f"Using saved voice '{args.narrator}'.")


def save_named_voice(name, voice):
    d = voices_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(voice["ref_audio"], d / "reference.wav")
    (d / "ref_text.txt").write_text(voice.get("ref_text") or "", encoding="utf-8")
    print(f"Voice saved as '{name}' ({d}). Next time just use: --narrator {name}")


def load_book_voice(args, workdir: Path):
    """Reuse the voice from an earlier run of this book when none is given."""
    f = workdir / "voice_settings.json"
    if args.ref or args.voice or args.narrator or not f.exists():
        return
    try:
        saved = json.loads(f.read_text(encoding="utf-8"))
    except Exception:
        return
    for k in VOICE_FIELDS:
        if k in saved:
            setattr(args, k, saved[k])
    print(f"Using the same voice as the previous run of this book "
          f"({'sample ' + Path(args.ref).name if args.ref else 'designed voice'}).")


def save_book_voice(args, workdir: Path):
    data = {k: getattr(args, k) for k in VOICE_FIELDS}
    if data["ref"]:
        data["ref"] = str(Path(data["ref"]).expanduser().resolve())
    (workdir / "voice_settings.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser(
        description="EPUB/TXT -> chaptered M4B audiobook with OmniVoice (Bulgarian by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("ebook", help=".epub or .txt file")
    ap.add_argument("-o", "--output", help="output .m4b (default: next to the ebook)")
    v = ap.add_argument_group("voice")
    v.add_argument("--ref", help="sample audio of the voice to clone (any format ffmpeg reads)")
    v.add_argument("--ref-text", help="exact transcript of the sample (recommended)")
    v.add_argument("--ref-max-secs", type=float, default=20, help="use at most this much of the sample")
    v.add_argument("--voice", help='without --ref: voice description, e.g. "female, low pitch"')
    v.add_argument("--design-model", default="k2-fsa/OmniVoice",
                   help="model used to design the voice from --voice (the fine-tune then clones it)")
    v.add_argument("--narrator", metavar="NAME", help="use a voice saved earlier with --save-voice")
    v.add_argument("--save-voice", metavar="NAME",
                   help="save the voice of this run (sample + text, or designed voice) under NAME")
    v.add_argument("--seed", type=int, default=1234, help="fixed seed for a consistent voice")
    g = ap.add_argument_group("engine")
    g.add_argument("--model", default=DEFAULT_MODEL)
    g.add_argument("--device", default="cpu", help="cpu, or cuda once ROCm PyTorch is installed")
    g.add_argument("--dtype", default="auto", choices=["auto", "float32", "float16", "bfloat16"],
                   help="auto = float32 on CPU, float16 on GPU (try float32 if GPU audio is garbled)")
    g.add_argument("--no-miopen", action="store_true",
                   help="ROCm: don't use MIOpen (try this if the GPU crashes or hangs)")
    g.add_argument("--threads", type=int, help="CPU threads for PyTorch (default: PyTorch decides)")
    g.add_argument("--gen-kwargs", default="{}",
                   help='extra JSON args passed to OmniVoice generate(), e.g. \'{"num_step": 16}\'')
    g.add_argument("--retries", type=int, default=2, help="regenerate chunks with suspicious length")
    t = ap.add_argument_group("text")
    t.add_argument("--chapters", help='only these chapters, e.g. "1-3,5" (numbers from --list)')
    t.add_argument("--max-chars", type=int, default=250, help="max characters per TTS chunk")
    t.add_argument("--min-chars", type=int, default=200, help="skip EPUB sections shorter than this")
    t.add_argument("--redo-chapters", metavar="SPEC",
                   help='regenerate these finished chapters, e.g. "1-6"')
    t.add_argument("--lexicon", help="pronunciation fixes file (word=replacement per line)")
    o = ap.add_argument_group("output / modes")
    o.add_argument("--bitrate", default="64k", help="AAC bitrate for the M4B")
    o.add_argument("--workdir", help="cache folder (default: <output>_work)")
    o.add_argument("--keep-chunks", action="store_true", help="keep per-chunk WAVs after a chapter is done")
    o.add_argument("--list", action="store_true", help="list detected chapters and exit")
    o.add_argument("--preview", type=int, nargs="?", const=6,
                   help="render the first N chunks of the first selected chapter to preview.wav")
    o.add_argument("--design-only", action="store_true",
                   help="without --ref: only create designed_voice.wav and exit")
    args = ap.parse_args()

    src = Path(args.ebook).expanduser()
    if not src.exists():
        die(f"File not found: {src}")
    book = load_book(src, args.min_chars)
    if not book.chapters:
        die("No text found. Try a lower --min-chars.")

    if args.list:
        print(f"{book.title} — {book.author or 'unknown author'}\n")
        total = 0
        for i, ch in enumerate(book.chapters, 1):
            total += ch.chars
            print(f"{i:3d}. {ch.title[:70]:70s} {ch.chars:7d} chars  ~{fmt_time(ch.chars / CHARS_PER_SEC)}")
        print(f"\nTotal ~{fmt_time(total / CHARS_PER_SEC)} of audio (rough estimate)")
        return

    need_ffmpeg()
    try:
        gen_kwargs = json.loads(args.gen_kwargs)
    except json.JSONDecodeError as e:
        die(f"--gen-kwargs is not valid JSON: {e}")
    out = Path(args.output).expanduser() if args.output else src.with_suffix(".m4b")
    workdir = Path(args.workdir).expanduser() if args.workdir else out.with_name(out.stem + "_work")
    workdir.mkdir(parents=True, exist_ok=True)

    if args.narrator:
        if args.ref:
            die("Use either --ref or --narrator, not both.")
        use_named_voice(args)
    load_book_voice(args, workdir)
    if args.ref and args.voice:
        print("Note: --ref given, ignoring --voice.")
    lexicon = load_lexicon(args.lexicon)
    selected = parse_chapters(args.chapters, len(book.chapters))
    if not selected:
        die("No chapters selected.")

    if args.device != "cpu":
        # ROCm/MIOpen: hide harmless warnings and skip the slow per-shape kernel search
        # (audio chunks all have different lengths). Override by setting these yourself.
        os.environ.setdefault("MIOPEN_LOG_LEVEL", "3")
        os.environ.setdefault("MIOPEN_FIND_MODE", "FAST")
    narr = Narrator(args.model, args.device, args.threads, gen_kwargs, args.dtype, args.no_miopen)
    voice, voice_key = prepare_voice(narr, args, workdir)
    save_book_voice(args, workdir)
    if args.save_voice:
        save_named_voice(args.save_voice, voice)
    if args.design_only:
        return

    if args.preview:
        ch = book.chapters[selected[0] - 1]
        render_preview(narr, ch, chapter_plan(ch, args.max_chars, lexicon), voice, args, workdir)
        return

    # Chapters rendered with a different voice must not be reused
    key_f = workdir / "voice_key.txt"
    old_key = key_f.read_text(encoding="utf-8") if key_f.exists() else None
    if old_key != voice_key:
        stale = sorted(workdir.glob("ch[0-9][0-9][0-9].wav"))
        if stale:
            print(f"Voice changed: discarding {len(stale)} chapter(s) made with the previous voice.")
            for f in stale:
                f.unlink()
        key_f.write_text(voice_key, encoding="utf-8")

    if args.redo_chapters:
        for idx in parse_chapters(args.redo_chapters, len(book.chapters)):
            f = workdir / f"ch{idx:03d}.wav"
            if f.exists():
                f.unlink()
                print(f"Chapter {idx} will be regenerated.")

    started = time.time()
    for idx in selected:
        ch = book.chapters[idx - 1]
        plan = chapter_plan(ch, args.max_chars, lexicon)
        if not any(k == "tts" for k, _ in plan):
            continue
        render_chapter(narr, idx, len(book.chapters), ch, plan, voice, voice_key, args, workdir)

    # Build the M4B from every chapter finished so far (this run or earlier runs),
    # so running --chapters 1-2 and then --chapters 3-6 gives a book with 1-6.
    files, titles, missing = [], [], []
    for idx, ch in enumerate(book.chapters, 1):
        f = workdir / f"ch{idx:03d}.wav"
        if f.exists():
            files.append(f)
            titles.append(ch.title)
        elif any(k == "tts" for k, _ in chapter_plan(ch, args.max_chars, lexicon)):
            missing.append(idx)
    if missing:
        print(f"\nNote: {len(files)} of {len(files) + len(missing)} chapters are done and included. "
              f"Not yet done: {compact_ranges(missing)}. Rerun to continue (done chapters are reused).")
    build_m4b(book, files, titles, out, workdir, args.bitrate)
    print(f"Total time: {fmt_time(time.time() - started)}")


if __name__ == "__main__":
    main()
