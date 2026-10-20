import json
import os
import sys
import time
from pathlib import Path

if not __package__ and Path(sys.path[0]).resolve() == Path(__file__).resolve().parent:
    # run by path: swap the package dir for its parent on sys.path, so our logging.py
    # does not shadow the stdlib logging module
    sys.path[0] = str(Path(__file__).resolve().parent.parent)

from book2audio.args import parse_arguments
from book2audio.config import CHARS_PER_SEC
from book2audio.exceptions import Book2AudioException, InputError
from book2audio.loaders import load_book
from book2audio.logging import error, log
from book2audio.output.audio import fmt_time
from book2audio.output.m4b import build_m4b, need_ffmpeg
from book2audio.output.render import render_chapter, render_preview
from book2audio.text.chunking import chapter_plan
from book2audio.text.lexicon import load_lexicon
from book2audio.tts.narrator import Narrator
from book2audio.tts.voice import (load_book_voice, prepare_voice, save_book_voice,
                                  save_named_voice, use_named_voice)
from book2audio.utils import compact_ranges, parse_chapters


def main():
    args = parse_arguments()
    try:
        run(args)
    except Book2AudioException as e:
        error(f"\nERROR: {e}")
        sys.exit(1)


def run(args):
    src = Path(args.ebook).expanduser()
    if not src.exists():
        raise InputError(f"File not found: {src}")
    book = load_book(src, args.min_chars)
    if not book.chapters:
        raise InputError("No text found. Try a lower --min-chars.")

    if args.list:
        log(f"{book.title} — {book.author or 'unknown author'}\n")
        total = 0
        for i, ch in enumerate(book.chapters, 1):
            total += ch.chars
            log(f"{i:3d}. {ch.title[:70]:70s} {ch.chars:7d} chars  ~{fmt_time(ch.chars / CHARS_PER_SEC)}")
        log(f"\nTotal ~{fmt_time(total / CHARS_PER_SEC)} of audio (rough estimate)")
        return

    need_ffmpeg()
    try:
        gen_kwargs = json.loads(args.gen_kwargs)
    except json.JSONDecodeError as e:
        raise InputError(f"--gen-kwargs is not valid JSON: {e}")
    out = Path(args.output).expanduser() if args.output else src.with_suffix(".m4b")
    workdir = Path(args.workdir).expanduser() if args.workdir else out.with_name(out.stem + "_work")
    workdir.mkdir(parents=True, exist_ok=True)

    if args.narrator:
        if args.ref:
            raise InputError("Use either --ref or --narrator, not both.")
        use_named_voice(args)
    load_book_voice(args, workdir)
    if args.ref and args.voice:
        log("Note: --ref given, ignoring --voice.")
    lexicon = load_lexicon(args.lexicon)
    selected = parse_chapters(args.chapters, len(book.chapters))
    if not selected:
        raise InputError("No chapters selected.")

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
            log(f"Voice changed: discarding {len(stale)} chapter(s) made with the previous voice.")
            for f in stale:
                f.unlink()
        key_f.write_text(voice_key, encoding="utf-8")

    if args.redo_chapters:
        for idx in parse_chapters(args.redo_chapters, len(book.chapters)):
            f = workdir / f"ch{idx:03d}.wav"
            if f.exists():
                f.unlink()
                log(f"Chapter {idx} will be regenerated.")

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
        log(f"\nNote: {len(files)} of {len(files) + len(missing)} chapters are done and included. "
              f"Not yet done: {compact_ranges(missing)}. Rerun to continue (done chapters are reused).")
    build_m4b(book, files, titles, out, workdir, args.bitrate)
    log(f"Total time: {fmt_time(time.time() - started)}")


if __name__ == "__main__":
    main()
