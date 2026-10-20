import re
import shutil
import subprocess
from pathlib import Path

import soundfile as sf

from book2audio import Book
from book2audio.exceptions import MissingDependency
from book2audio.logging import log
from book2audio.output.audio import fmt_time
from book2audio.workdir import WorkDir


def need_ffmpeg():
    if not shutil.which("ffmpeg"):
        raise MissingDependency("ffmpeg not found. Install it, e.g.: sudo apt install ffmpeg")


def run_ffmpeg(args):
    subprocess.run(["ffmpeg", "-y", "-hide_banner", "-loglevel", "error", *args], check=True)


def detect_pauses(path):
    """Start times (s) of pauses in an audio file, via ffmpeg silencedetect."""
    r = subprocess.run(["ffmpeg", "-hide_banner", "-i", str(path), "-af",
                        "silencedetect=noise=-35dB:d=0.3", "-f", "null", "-"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    return [float(x) for x in re.findall(r"silence_start: ([\d.]+)", r.stderr)]


def ffmeta_escape(s: str) -> str:
    return re.sub(r"([=;#\\\n])", r"\\\1", s or "")


class M4bBuilder:
    """Concatenates finished chapter WAVs into one chaptered M4B."""

    def __init__(self, book: Book, workdir: WorkDir, bitrate: str):
        self.book = book
        self.workdir = workdir
        self.bitrate = bitrate

    def build(self, files, titles, out: Path):
        lst = self.workdir.concat_list
        with open(lst, "w", encoding="utf-8") as fh:
            for p in files:
                fh.write("file '" + str(Path(p).resolve()).replace("'", "'\\''") + "'\n")

        e = ffmeta_escape
        lines = [";FFMETADATA1", f"title={e(self.book.title)}", f"album={e(self.book.title)}",
                 f"artist={e(self.book.author)}", f"album_artist={e(self.book.author)}", "genre=Audiobook"]
        t = 0
        for p, title in zip(files, titles):
            info = sf.info(str(p))
            dur = int(round(info.frames * 1000 / info.samplerate))
            lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={t}", f"END={t + dur}",
                      f"title={e(title)}"]
            t += dur
        meta = self.workdir.ffmeta
        meta.write_text("\n".join(lines) + "\n", encoding="utf-8")

        cmd = ["-f", "concat", "-safe", "0", "-i", str(lst), "-i", str(meta)]
        cover = None
        if self.book.cover:
            cover = self.workdir.cover(self.book.cover_ext)
            cover.write_bytes(self.book.cover)
            cmd += ["-i", str(cover)]
        cmd += ["-map", "0:a", "-map_metadata", "1", "-map_chapters", "1"]
        if cover:
            cmd += ["-map", "2:v", "-c:v", "copy", "-disposition:v:0", "attached_pic"]
        cmd += ["-c:a", "aac", "-b:a", self.bitrate, "-ac", "1", "-movflags", "+faststart",
                "-f", "mp4", str(out)]
        log(f"\nEncoding {out.name} ({fmt_time(t / 1000)}) ...", flush=True)
        try:
            run_ffmpeg(cmd)
        except subprocess.CalledProcessError:
            if not cover:
                raise
            log("Cover could not be embedded, retrying without it ...")
            cmd = [x for x in cmd]  # rebuild without cover input/mapping
            i = cmd.index(str(cover))
            del cmd[i - 1:i + 1]
            j = cmd.index("2:v")
            del cmd[j - 1:j + 5]
            run_ffmpeg(cmd)
        log(f"Done: {out}")
