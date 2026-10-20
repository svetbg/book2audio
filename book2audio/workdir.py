import hashlib
import json
from pathlib import Path

from book2audio.logging import log

VOICE_FIELDS = ("ref", "ref_text", "voice", "seed", "design_model", "ref_max_secs")


class WorkDir:
    """The cache folder of one book: the only place that knows what its files are called."""

    @classmethod
    def for_output(cls, out: Path) -> "WorkDir":
        """The default work folder of an output file: next to it, named <stem>_work."""
        return cls(out.with_name(out.stem + "_work"))

    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)
        self.designed_wav = self.path / "designed_voice.wav"
        self.designed_json = self.path / "designed_voice.json"
        self.reference_full = self.path / "reference_full.wav"
        self.reference = self.path / "reference.wav"
        self.preview_wav = self.path / "preview.wav"
        self.concat_list = self.path / "concat.txt"
        self.ffmeta = self.path / "chapters.ffmeta"
        self._voice_key = self.path / "voice_key.txt"
        self._voice_settings = self.path / "voice_settings.json"

    def chapter_wav(self, idx: int) -> Path:
        return self.path / f"ch{idx:03d}.wav"

    def chapter_tmp(self, idx: int) -> Path:
        return self.path / f"ch{idx:03d}.tmp.wav"

    def chunk_dir(self, idx: int) -> Path:
        return self.path / f"ch{idx:03d}_chunks"

    def chunk_file(self, idx: int, k: int, voice_key: str, text: str) -> Path:
        """Chunk WAV keyed by voice and text, so it survives reordering."""
        h = hashlib.md5((voice_key + "|" + text).encode("utf-8")).hexdigest()[:10]
        cdir = self.chunk_dir(idx)
        legacy = cdir / f"{k:04d}_{h}.wav"         # chunks from older script versions
        return legacy if legacy.exists() else cdir / f"{h}.wav"

    @staticmethod
    def chunk_tmp(chunk: Path) -> Path:
        return chunk.with_name(chunk.stem + ".tmp.wav")

    def cover(self, ext: str) -> Path:
        return self.path / f"cover.{ext}"

    def finished_chapters(self):
        return sorted(self.path.glob("ch[0-9][0-9][0-9].wav"))

    def read_voice_key(self):
        return self._voice_key.read_text(encoding="utf-8") if self._voice_key.exists() else None

    def write_voice_key(self, key: str):
        self._voice_key.write_text(key, encoding="utf-8")

    def load_voice_settings(self, args):
        """Reuse the voice from an earlier run of this book when none is given."""
        f = self._voice_settings
        if args.ref or args.voice or args.narrator or not f.exists():
            return
        try:
            saved = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            return
        for k in VOICE_FIELDS:
            if k in saved:
                setattr(args, k, saved[k])
        log(f"Using the same voice as the previous run of this book "
              f"({'sample ' + Path(args.ref).name if args.ref else 'designed voice'}).")

    def save_voice_settings(self, args):
        data = {k: getattr(args, k) for k in VOICE_FIELDS}
        if data["ref"]:
            data["ref"] = str(Path(data["ref"]).expanduser().resolve())
        self._voice_settings.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
