import json
import time
from pathlib import Path

from book2audio.config import CHARS_PER_SEC
from book2audio.exceptions import InputError
from book2audio.loaders import for_path
from book2audio.logging import log
from book2audio.output.audio import fmt_time
from book2audio.output.m4b import M4bBuilder, need_ffmpeg
from book2audio.output.render import ChapterRenderer
from book2audio.text.chunking import Chunker
from book2audio.text.lexicon import Lexicon
from book2audio.tts.narrator import Narrator, set_miopen_env
from book2audio.tts.voice import DesignedVoice, SampleVoice, VoiceLibrary
from book2audio.utils import compact_ranges, parse_chapters
from book2audio.workdir import WorkDir


class Book2Audio:
    """The app: wires loader, voice, narrator, renderer and builder together from the options."""

    def __init__(self, args):
        self.args = args

    def run(self):
        args = self.args
        src = Path(args.ebook).expanduser()
        if not src.exists():
            raise InputError(f"File not found: {src}")
        book = for_path(src, args.min_chars).load()
        if not book.chapters:
            raise InputError("No text found. Try a lower --min-chars.")

        if args.list:
            self._list(book)
            return

        need_ffmpeg()
        try:
            gen_kwargs = json.loads(args.gen_kwargs)
        except json.JSONDecodeError as e:
            raise InputError(f"--gen-kwargs is not valid JSON: {e}")
        out = Path(args.output).expanduser() if args.output else src.with_suffix(".m4b")
        workdir = (WorkDir(Path(args.workdir).expanduser()) if args.workdir
                   else WorkDir.for_output(out))

        library = VoiceLibrary()
        if args.narrator:
            if args.ref:
                raise InputError("Use either --ref or --narrator, not both.")
            saved = library.get(args.narrator)
            args.ref, args.ref_text = saved["ref_audio"], saved["ref_text"]
        workdir.load_voice_settings(args)
        if args.ref and args.voice:
            log("Note: --ref given, ignoring --voice.")
        chunker = Chunker(args.max_chars, Lexicon.from_file(args.lexicon))
        selected = parse_chapters(args.chapters, len(book.chapters))
        if not selected:
            raise InputError("No chapters selected.")

        set_miopen_env(args.device)
        narr = Narrator(args.model, args.device, args.threads, gen_kwargs, args.dtype, args.no_miopen)
        voice, voice_key = self._voice().prepare(narr, workdir)
        workdir.save_voice_settings(args)
        if args.save_voice:
            library.save(args.save_voice, voice)
        if args.design_only:
            return

        renderer = ChapterRenderer(narr, voice, voice_key, workdir, args.seed, args.retries,
                                   args.keep_chunks)
        if args.preview:
            ch = book.chapters[selected[0] - 1]
            renderer.preview(chunker.plan(ch), args.preview)
            return

        # Chapters rendered with a different voice must not be reused
        if workdir.read_voice_key() != voice_key:
            stale = workdir.finished_chapters()
            if stale:
                log(f"Voice changed: discarding {len(stale)} chapter(s) made with the previous voice.")
                for f in stale:
                    f.unlink()
            workdir.write_voice_key(voice_key)

        if args.redo_chapters:
            for idx in parse_chapters(args.redo_chapters, len(book.chapters)):
                f = workdir.chapter_wav(idx)
                if f.exists():
                    f.unlink()
                    log(f"Chapter {idx} will be regenerated.")

        started = time.time()
        for idx in selected:
            ch = book.chapters[idx - 1]
            plan = chunker.plan(ch)
            if not any(k == "tts" for k, _ in plan):
                continue
            renderer.render(idx, len(book.chapters), ch, plan)

        # Build the M4B from every chapter finished so far (this run or earlier runs),
        # so running --chapters 1-2 and then --chapters 3-6 gives a book with 1-6.
        files, titles, missing = [], [], []
        for idx, ch in enumerate(book.chapters, 1):
            f = workdir.chapter_wav(idx)
            if f.exists():
                files.append(f)
                titles.append(ch.title)
            elif any(k == "tts" for k, _ in chunker.plan(ch)):
                missing.append(idx)
        if missing:
            log(f"\nNote: {len(files)} of {len(files) + len(missing)} chapters are done and included. "
                  f"Not yet done: {compact_ranges(missing)}. Rerun to continue (done chapters are reused).")
        M4bBuilder(book, workdir, args.bitrate).build(files, titles, out)
        log(f"Total time: {fmt_time(time.time() - started)}")

    def _voice(self):
        a = self.args
        if a.ref:
            return SampleVoice(a.ref, a.ref_text, a.seed, a.ref_max_secs)
        return DesignedVoice(a.voice, a.seed, a.design_model, a.model)

    @staticmethod
    def _list(book):
        log(f"{book.title} — {book.author or 'unknown author'}\n")
        total = 0
        for i, ch in enumerate(book.chapters, 1):
            total += ch.chars
            log(f"{i:3d}. {ch.title[:70]:70s} {ch.chars:7d} chars  ~{fmt_time(ch.chars / CHARS_PER_SEC)}")
        log(f"\nTotal ~{fmt_time(total / CHARS_PER_SEC)} of audio (rough estimate)")
