import json
import os
import re
import shutil
from pathlib import Path

import soundfile as sf

from book2audio.config import DESIGN_SENTENCE
from book2audio.exceptions import InputError, VoiceError
from book2audio.logging import log
from book2audio.output.audio import normalize, trim_silence
from book2audio.output.m4b import detect_pauses, run_ffmpeg
from book2audio.tts.narrator import Narrator


def prepare_voice(narr, args, workdir: Path):
    """Returns (voice kwargs for generate, cache key)."""
    if args.ref:
        src = Path(args.ref).expanduser()
        if not src.exists():
            raise InputError(f"Reference file not found: {src}")
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
                raise VoiceError(f"The sample is {dur:.1f}s long, but only the first {args.ref_max_secs:g}s "
                    f"are used.\nThen --ref-text contains words that are NOT in the reference audio, "
                    f"and the model speaks those words inside the book.\n"
                    f"Cut the sample at a pause and give only the words spoken up to that point:\n"
                    f"{hint}  ffmpeg -i \"{src}\" -to {cut} -c copy sample_cut.wav\n"
                    f"Or use the whole sample with --ref-max-secs {int(dur) + 1} (slower on CPU).")
            log(f"Note: using only the first {args.ref_max_secs:g}s of the {dur:.1f}s sample.")

        ref = workdir / "reference.wav"
        run_ffmpeg(["-i", str(full), "-t", str(args.ref_max_secs), str(ref)])

        if ref_text:
            used = min(dur, args.ref_max_secs)
            rate = len(re.findall(r"[^\W_]", ref_text)) / max(used, 0.1)
            if re.search(r"\d", ref_text):
                log("Tip: write numbers in --ref-text as words, exactly as they are spoken "
                      "(e.g. 'хиляда осемстотин шейсет и шеста').")
            if not 8 <= rate <= 20:
                log(f"Warning: --ref-text has ~{rate:.0f} letters per second of audio "
                      f"(normal speech is ~12-16). It probably doesn't match the clip exactly, "
                      f"which can make words from the sample appear in the book.")
            if not re.search(r"[.!?…][\"»”“'’)]*$", ref_text):
                ref_text += "."
        else:
            log("Note: no --ref-text given. Giving the exact transcript of the sample "
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
        log("Voice settings changed since designed_voice.wav was made: designing a new one.")
        designed.unlink()
    if not designed.exists():
        # The Bulgarian fine-tune may ignore voice descriptions, so by default the
        # voice is designed with the base OmniVoice model and only then cloned.
        dn = narr if want["model"] == args.model else Narrator(
            want["model"], args.device, args.threads, {}, args.dtype, args.no_miopen)
        log(f"No sample given: designing a voice ({args.voice or 'automatic'}) "
              f"with {want['model']} ...", flush=True)
        a = dn.generate(DESIGN_SENTENCE, args.seed, instruct=args.voice)
        sf.write(designed, normalize(trim_silence(a, dn.sr)), dn.sr, subtype="PCM_16")
        meta_f.write_text(json.dumps(want, ensure_ascii=False), encoding="utf-8")
        if dn is not narr:
            del dn
            import gc
            gc.collect()
        log(f"Saved {designed}\n  Listen to it. If you don't like it, rerun with "
              f"another --seed and/or --voice (it is regenerated automatically).")
    key = f"designed:{args.voice}:{args.seed}:{designed.stat().st_mtime}"
    return {"ref_audio": str(designed), "ref_text": DESIGN_SENTENCE}, key


VOICE_FIELDS = ("ref", "ref_text", "voice", "seed", "design_model", "ref_max_secs")


def voices_dir() -> Path:
    base = Path(os.environ.get("XDG_DATA_HOME", "~/.local/share")).expanduser()
    return base / "book2audio" / "voices"


def use_named_voice(args):
    d = voices_dir() / args.narrator
    ref, txt = d / "reference.wav", d / "ref_text.txt"
    if not ref.exists():
        have = sorted(p.name for p in voices_dir().glob("*") if (p / "reference.wav").exists())
        raise InputError(f"No saved voice named '{args.narrator}'. "
            f"Saved voices: {', '.join(have) if have else 'none yet (use --save-voice NAME)'}")
    args.ref = str(ref)
    args.ref_text = txt.read_text(encoding="utf-8").strip() if txt.exists() else None
    log(f"Using saved voice '{args.narrator}'.")


def save_named_voice(name, voice):
    d = voices_dir() / name
    d.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(voice["ref_audio"], d / "reference.wav")
    (d / "ref_text.txt").write_text(voice.get("ref_text") or "", encoding="utf-8")
    log(f"Voice saved as '{name}' ({d}). Next time just use: --narrator {name}")


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
    log(f"Using the same voice as the previous run of this book "
          f"({'sample ' + Path(args.ref).name if args.ref else 'designed voice'}).")


def save_book_voice(args, workdir: Path):
    data = {k: getattr(args, k) for k in VOICE_FIELDS}
    if data["ref"]:
        data["ref"] = str(Path(data["ref"]).expanduser().resolve())
    (workdir / "voice_settings.json").write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
