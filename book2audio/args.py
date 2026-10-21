import argparse

from book2audio import __version__
from book2audio.config import DEFAULT_DESIGN_MODEL, DEFAULT_MODEL


def parse_arguments() -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        prog="book2audio",
        description="EPUB/TXT -> chaptered M4B audiobook with OmniVoice (Bulgarian by default).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    ap.add_argument("-v", "--version", action="version", version=f"book2audio {__version__}")
    ap.add_argument("ebook", help=".epub or .txt file")
    ap.add_argument("-o", "--output", help="output .m4b (default: next to the ebook)")
    v = ap.add_argument_group("voice")
    v.add_argument("--ref", help="sample audio of the voice to clone (any format ffmpeg reads)")
    v.add_argument("--ref-text", help="exact transcript of the sample (recommended)")
    v.add_argument("--ref-max-secs", type=float, default=20, help="use at most this much of the sample")
    v.add_argument("--voice", help='without --ref: voice description, e.g. "female, low pitch"')
    v.add_argument("--design-model", default=DEFAULT_DESIGN_MODEL,
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
    return ap.parse_args()
