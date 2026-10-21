# book2audio

Turn an EPUB or TXT ebook into a chaptered **M4B audiobook** using [OmniVoice](https://huggingface.co/k2-fsa/OmniVoice) text-to-speech. The default model is `preslaff/omnivoice-bg`, a Bulgarian fine-tune, so it is aimed at Bulgarian books out of the box (another OmniVoice model can be chosen with `--model`).

What it does:

- reads an `.epub` (chapters are the spine documents with at least `--min-chars` characters (the table of contents only supplies their titles), cover image, title and author) or a `.txt` file (cp1251 or UTF-8, chapters found by lines such as "Глава 1" / "Chapter 1");
- splits every chapter into sentence-sized chunks and narrates them with OmniVoice, in the voice of a sample you supply or of a voice designed from a short description;
- caches every chunk and chapter, so an interrupted run resumes where it stopped;
- joins the chapters into one `.m4b` with chapter markers, cover and metadata via ffmpeg.

## Requirements

- Python 3.10 or newer
- [ffmpeg](https://ffmpeg.org/) on the `PATH` (`sudo apt install ffmpeg`)
- PyTorch, installed separately (see below), unless the model runs on a [remote server](#remote-gpu-server)
- A Hugging Face account that has accepted the conditions of the gated model

## Installation

Two kinds of install:

- **Client only**: reads the book and builds the M4B, and sends narration to a [remote server](#remote-gpu-server). No PyTorch: skip step 1 and run `pip install .`.
- **Local narration, and the server**: runs the model on this machine. Install PyTorch first (step 1), then `pip install ".[local]"`.

Use a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1. PyTorch (first, and separately; not needed for a client-only install)

PyTorch is intentionally **not** a dependency of this package: the right build depends on your hardware. Install one of these *before* installing book2audio with the `local` extra: omnivoice depends on torch, so installing it first pulls the default multi-GB CUDA build from PyPI.

CPU only:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cpu
```

AMD GPU (ROCm). Pick the ROCm version that matches your system from <https://pytorch.org/get-started/locally/> and substitute it for `rocm6.2`:

```bash
pip install torch torchaudio --index-url https://download.pytorch.org/whl/rocm6.2
```

For a ROCm build, check that PyTorch sees the GPU:

```bash
python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"
```

This should print a HIP version and `True`. (`torch.cuda` is the name PyTorch uses for ROCm devices too, which is why the option is `--device cuda`.)

### 2. book2audio

```bash
pip install .            # client only: the book2audio command, no PyTorch
pip install ".[local]"     # local narration and book2audio-server: adds omnivoice (PyTorch first, step 1)
pip install -e ".[local,test]"   # or: editable install with omnivoice and pytest, for development
```

The dependencies (numpy, soundfile, ebooklib, beautifulsoup4, and omnivoice with `[local]`) are listed in `pyproject.toml` and installed by pip automatically.

**Existing environments keep working.** omnivoice used to be installed with `pip install .`. An environment where it is already installed still has it, and nothing there needs to change. Only a *new* environment needs `.[local]` for local narration; without it, narrating locally stops with "OmniVoice is not installed".

### 3. Hugging Face access

Only the Bulgarian fine-tune (`preslaff/omnivoice-bg`) is gated; `k2-fsa/OmniVoice`, used for designed voices, is not. Once per machine:

1. Open <https://huggingface.co/preslaff/omnivoice-bg> while logged in, and accept the conditions.
2. Log in from the terminal: `hf auth login` (or `huggingface-cli login` with older versions) and paste an access token.

### Running

Both forms are equivalent:

```bash
book2audio book.epub ...
python book2audio.py book.epub ...     # wrapper, works straight from a checkout
```

`--help`, `--list` and `--version` work without PyTorch or OmniVoice installed; the heavy libraries are only imported when narration starts.

## Quick start

```bash
# 1. check how the book is split into chapters
book2audio book.epub --list

# 2. narrate it with a voice sample
book2audio book.epub --ref narrator.wav --ref-text "exact words spoken in the sample"
```

The result is `book.m4b` next to the ebook (change with `-o`), and the cache lives in `book_work/` (change with `--workdir`).

## Voices

There are two ways to choose the narrator.

### 1. Voice sample (cloning)

```bash
book2audio book.epub --ref narrator.wav --ref-text "exact words spoken in the sample"
```

- `--ref` is any audio file ffmpeg can read.
- `--ref-text` is the **exact** transcript of what is spoken in the sample. It is optional but strongly recommended; without it, cloning is usually worse.
- `--ref-max-secs` (default 20) is how much of the sample is used. If the sample is longer *and* you pass `--ref-text`, book2audio stops with an error, because the transcript would contain words the model never hears. It prints the pauses it found in the sample, and an ffmpeg command to cut the sample at a pause. Alternatively raise `--ref-max-secs` to cover the whole sample (slower on CPU).

#### Making a sample from an existing recording

A good sample is 10-20 seconds of one narrator, with no music or background noise, cut at pauses between sentences. To take one from an audiobook you already have, and get its transcript:

```bash
ffmpeg -ss 00:12:31.8 -to 00:12:45.2 -i book.m4b -vn \
  -af "loudnorm=I=-20:TP=-2" -ac 1 -ar 24000 sample.wav
```

- `-ss 00:12:31.8` / `-to 00:12:45.2`: start and end of the excerpt (`HH:MM:SS.s`), here 13.4 seconds. Placed before `-i`, they make ffmpeg seek straight to the start instead of decoding everything before it. Pick both points in a pause, so no word is cut in half.
- `-i book.m4b`: the input; any format ffmpeg reads.
- `-vn`: drop the video stream (an M4B's cover art is one), keep only audio.
- `-af "loudnorm=I=-20:TP=-2"`: EBU R128 loudness normalisation to an integrated loudness of -20 LUFS with true peaks at most -2 dBTP, so a quiet or loud recording becomes a sample of even, moderate volume without clipping.
- `-ac 1 -ar 24000`: mono, 24 kHz, the format OmniVoice works in. book2audio converts `--ref` to this itself, so this only makes `sample.wav` sound exactly as the model will hear it.
- `sample.wav`: the output file; the `.wav` extension selects uncompressed PCM.

```bash
pip install openai-whisper
whisper sample.wav --language bg --model medium
```

- `pip install openai-whisper`: OpenAI's Whisper speech recognition, installed into the same virtual environment. It uses the PyTorch you already installed, and ffmpeg to read audio.
- `whisper sample.wav`: transcribes the sample, prints the text with timestamps, and writes it to `sample.txt` (plus `.srt`, `.vtt`, `.tsv`, `.json`) in the current folder.
- `--language bg`: the audio is Bulgarian; skips language detection, which can guess wrong on a short clip.
- `--model medium`: the model size (`tiny`, `base`, `small`, `medium`, `large`; bigger is slower and more accurate). `medium` is about 1.5 GB, downloaded once to `~/.cache/whisper/`.

Whisper is not always exact: listen to the sample while reading `sample.txt`, fix any wrong words, and write numbers out as words (see [Troubleshooting](#troubleshooting)). Then use it:

```bash
book2audio book.epub --ref sample.wav --ref-text "$(cat sample.txt)"
```

### 2. Designed voice (no sample)

```bash
book2audio book.epub --voice "female, low pitch" --design-only   # audition the voice
book2audio book.epub --voice "female, low pitch"                 # narrate the book
```

A voice is designed once from the description, saved as `designed_voice.wav` in the work folder, and then used as the reference for the whole book so it stays consistent.

- `--voice` is the description (e.g. `"male, low pitch"`). Without it, an automatic voice is used.
- `--design-model` (default `k2-fsa/OmniVoice`) is the model that designs the voice. The Bulgarian fine-tune may ignore voice descriptions, so the base model designs the voice and the fine-tune then clones it.
- `--design-only` creates `designed_voice.wav` and exits, so you can listen to it. Not happy? Re-run with another `--seed` and/or `--voice`; the voice is regenerated automatically when either changes.
- `--seed` (default 1234) makes the result reproducible.

### Saved voices

Keep a voice you like and reuse it for other books:

```bash
book2audio book.epub --ref narrator.wav --ref-text "..." --save-voice anna   # save while narrating
book2audio other.epub --narrator anna                                         # reuse later
```

`--save-voice NAME` stores the sample and its text (or the designed voice) in `$XDG_DATA_HOME/book2audio/voices/NAME/` (default `~/.local/share/book2audio/voices/NAME/`). `--narrator NAME` loads it; it cannot be combined with `--ref`.

A book also remembers the voice of its previous run (`voice_settings.json` in the work folder), so re-running a book without any voice option continues with the same voice.

## Recommended workflow

```bash
book2audio book.epub --list                                   # 1. check chapter detection
book2audio book.epub --ref narrator.wav --preview             # 2. first ~6 chunks -> preview.wav
book2audio book.epub --ref narrator.wav --chapters 1-2        # 3. test run on two chapters
book2audio book.epub --ref narrator.wav                       # 4. the whole book -> book.m4b
```

- `--list` prints the detected chapters with their size and a rough duration estimate. If a TXT or EPUB is split badly, adjust `--min-chars` (EPUB sections shorter than this are skipped, default 200).
- `--preview [N]` renders the first N chunks (default 6) of the first selected chapter to `preview.wav` in the work folder and prints the real-time factor, so you can judge the voice and the speed.
- `--chapters "1-3,5"` limits narration to some chapters (numbers from `--list`). The M4B is built from **every chapter finished so far**, so running `--chapters 1-2` and later `--chapters 3-6` gives a book with chapters 1-6.

## Resuming and caching

Re-running the same command resumes the work. In the work folder (`<output>_work/`, or `--workdir`):

| File | Meaning |
| --- | --- |
| `chNNN.wav` | a finished chapter; skipped on later runs |
| `chNNN_chunks/` | per-chunk WAVs of an unfinished chapter, named by a hash of voice and text (removed when the chapter is done, unless `--keep-chunks`) |
| `voice_key.txt`, `voice_settings.json` | identity and settings of the voice used |
| `designed_voice.wav` / `.json` | the designed voice and the settings it was made with |
| `reference.wav`, `reference_full.wav` | the converted voice sample |
| `preview.wav` | output of `--preview` |
| `cover.jpg` / `cover.png` | the EPUB cover, extracted for the M4B |
| `*.tmp.wav` | temporary files of a chapter or chunk being written; safe to delete after an interruption |
| `concat.txt`, `chapters.ffmeta` | ffmpeg inputs for the final M4B |

What invalidates the cache:

- **Changing the voice** (a different sample, `--ref-text`, `--seed`, `--ref-max-secs`, or a regenerated designed voice) discards the finished chapters made with the previous voice (the message "Voice changed: discarding N chapter(s)" is printed). Chunks are keyed by voice and text, so they are never reused across voices.
- **`--redo-chapters SPEC`** (e.g. `--redo-chapters 1-6`) deletes those finished chapters so they are regenerated; chunks that are still cached are reused. Delete the `chNNN_chunks/` folder as well to force new audio for the text.
- Changing the **lexicon** or `--max-chars` changes the text of chunks, so affected chunks are narrated again, but already finished chapters are *not* touched unless you use `--redo-chapters`.

## Pronunciation lexicon

`--lexicon FILE` fixes readings and stress. One rule per line: `word=replacement`, or `word<TAB>replacement`. Lines starting with `#` and blank lines are ignored. Matching is case-insensitive on whole words, and a capital first letter in the text is kept in the replacement.

```text
# stress marks: put U+0301 (combining acute accent) after the stressed vowel
бутона=буто́на
замък	за́мък
```

## GPU / ROCm

Narration runs on the CPU by default. With a ROCm build of PyTorch:

```bash
book2audio book.epub --ref narrator.wav --device cuda
```

- `--device` is `cpu` (default) or `cuda` (also used for ROCm).
- `--dtype` is `auto` (float32 on CPU, float16 on GPU), `float32`, `float16` or `bfloat16`. If GPU audio is garbled, use `--dtype float32`.
- `--no-miopen` disables MIOpen (ROCm's cuDNN equivalent). Try it if the GPU crashes or hangs.
- With `--device` other than `cpu`, `MIOPEN_LOG_LEVEL=3` (hide harmless warnings) and `MIOPEN_FIND_MODE=FAST` (skip the slow per-shape kernel search; audio chunks all have different lengths) are set automatically, unless you have set them yourself.
- `--threads N` sets the number of CPU threads for PyTorch.
- `--gen-kwargs '{"num_step": 16}'` passes extra JSON arguments to OmniVoice's `generate()`.

## Remote GPU server

Run OmniVoice on a GPU machine (the *server*) and book2audio anywhere else (the *client*). The client keeps everything it does today (reading the book, chunking, the cache in the work folder, retries, building the M4B); only the narration of each chunk runs on the server. Without `--server` nothing changes.

```bash
# on the GPU machine (PyTorch and `pip install ".[local]"`, see Installation)
book2audio-server --device cuda

# on the client (through an SSH tunnel, see Security below)
ssh -N -L 8765:localhost:8765 gpu-box &
book2audio book.epub --ref sample.wav --ref-text "..." --server http://localhost:8765
```

- The client needs no PyTorch: a plain `pip install .` is enough.
- `book2audio-server` takes `--host` (default `127.0.0.1`), `--port` (default `8765`), `--device`, `--dtype`, `--no-miopen` and `--threads` (same meaning as for `book2audio`), and `--model M` (repeatable). It loads the models at startup, by default the `--model` and `--design-model` defaults of the client. The models listed are the only ones a client may use; any other is refused, so a client cannot make the server download arbitrary repositories.
- The client takes host and port from `--server URL`. With a different `--port` on the server, change the URL (and the tunnel) to match.
- `--device`, `--dtype`, `--no-miopen` and `--threads` belong to the server; given with `--server` they are ignored, with a note.
- The cache keys and the work folder are the same as for a local run, so you can switch between local narration and `--server` partway through a book and everything already finished is reused.
- The server handles one request at a time. If the network drops, the run stops with an error, and running it again resumes from the cache.
- The reference sample is uploaded once per run and kept on the server in a temporary folder until it stops. A server restart loses it, and the client uploads it again by itself.

### Security

The server has **no authentication and no encryption**, and `--gen-kwargs` from a client are passed straight to OmniVoice's `generate()`. It listens on `127.0.0.1` by default, so only the machine itself can reach it. Reach it through an SSH tunnel (`ssh -L`, as above), or use `--host 0.0.0.0` only on a network you trust. Do not expose it to the internet.

## Output

- A single `.m4b` (MP4 container, AAC mono, `--bitrate`, default 64k, `+faststart`).
- One chapter marker per book chapter, titled from the EPUB table of contents, else the first h1-h3 heading, else the first paragraph if it is at most 80 characters, else "Глава N".
- Metadata: title and album from the book title, artist and album artist from the author, genre `Audiobook`.
- The EPUB cover is embedded as cover art. If ffmpeg cannot embed it, the book is built without it.
- Each chapter starts with 0.8 s and ends with 2 s of silence; pauses between chunks, paragraphs and scene breaks (`* * *`) are added automatically.
- Chunks of suspicious length (too short or too long for their text) are regenerated with another seed, up to `--retries` times (default 2).

## Option reference

Output of `book2audio --help`:

```text
usage: book2audio [-h] [-v] [-o OUTPUT] [--ref REF] [--ref-text REF_TEXT]
                  [--ref-max-secs REF_MAX_SECS] [--voice VOICE]
                  [--design-model DESIGN_MODEL] [--narrator NAME]
                  [--save-voice NAME] [--seed SEED] [--model MODEL]
                  [--server URL] [--device DEVICE]
                  [--dtype {auto,float32,float16,bfloat16}] [--no-miopen]
                  [--threads THREADS] [--gen-kwargs GEN_KWARGS]
                  [--retries RETRIES] [--chapters CHAPTERS]
                  [--max-chars MAX_CHARS] [--min-chars MIN_CHARS]
                  [--redo-chapters SPEC] [--lexicon LEXICON]
                  [--bitrate BITRATE] [--workdir WORKDIR] [--keep-chunks]
                  [--list] [--preview [PREVIEW]] [--design-only]
                  ebook

EPUB/TXT -> chaptered M4B audiobook with OmniVoice (Bulgarian by default).

positional arguments:
  ebook                 .epub or .txt file

options:
  -h, --help            show this help message and exit
  -v, --version         show program's version number and exit
  -o OUTPUT, --output OUTPUT
                        output .m4b (default: next to the ebook) (default:
                        None)

voice:
  --ref REF             sample audio of the voice to clone (any format ffmpeg
                        reads) (default: None)
  --ref-text REF_TEXT   exact transcript of the sample (recommended) (default:
                        None)
  --ref-max-secs REF_MAX_SECS
                        use at most this much of the sample (default: 20)
  --voice VOICE         without --ref: voice description, e.g. "female, low
                        pitch" (default: None)
  --design-model DESIGN_MODEL
                        model used to design the voice from --voice (the fine-
                        tune then clones it) (default: k2-fsa/OmniVoice)
  --narrator NAME       use a voice saved earlier with --save-voice (default:
                        None)
  --save-voice NAME     save the voice of this run (sample + text, or designed
                        voice) under NAME (default: None)
  --seed SEED           fixed seed for a consistent voice (default: 1234)

engine:
  --model MODEL
  --server URL          run the model on a book2audio-server at URL instead of
                        locally (default: None)
  --device DEVICE       cpu, or cuda once ROCm PyTorch is installed (default:
                        cpu)
  --dtype {auto,float32,float16,bfloat16}
                        auto = float32 on CPU, float16 on GPU (try float32 if
                        GPU audio is garbled) (default: auto)
  --no-miopen           ROCm: don't use MIOpen (try this if the GPU crashes or
                        hangs) (default: False)
  --threads THREADS     CPU threads for PyTorch (default: PyTorch decides)
                        (default: None)
  --gen-kwargs GEN_KWARGS
                        extra JSON args passed to OmniVoice generate(), e.g.
                        '{"num_step": 16}' (default: {})
  --retries RETRIES     regenerate chunks with suspicious length (default: 2)

text:
  --chapters CHAPTERS   only these chapters, e.g. "1-3,5" (numbers from
                        --list) (default: None)
  --max-chars MAX_CHARS
                        max characters per TTS chunk (default: 250)
  --min-chars MIN_CHARS
                        skip EPUB sections shorter than this (default: 200)
  --redo-chapters SPEC  regenerate these finished chapters, e.g. "1-6"
                        (default: None)
  --lexicon LEXICON     pronunciation fixes file (word=replacement per line)
                        (default: None)

output / modes:
  --bitrate BITRATE     AAC bitrate for the M4B (default: 64k)
  --workdir WORKDIR     cache folder (default: <output>_work) (default: None)
  --keep-chunks         keep per-chunk WAVs after a chapter is done (default:
                        False)
  --list                list detected chapters and exit (default: False)
  --preview [PREVIEW]   render the first N chunks of the first selected
                        chapter to preview.wav (default: None)
  --design-only         without --ref: only create designed_voice.wav and exit
                        (default: False)
```

## Troubleshooting

- **Words from the sample appear in the book.** `--ref-text` does not match the audio exactly (or the sample is longer than `--ref-max-secs`, so part of the transcript is not heard). Give the exact words, cut the sample at a pause, or raise `--ref-max-secs`. book2audio warns when the text has an implausible number of letters per second.
- **Numbers in `--ref-text`.** Write them as words, exactly as spoken (e.g. "хиляда осемстотин шейсет и шеста").
- **Garbled audio on the GPU.** Use `--dtype float32`.
- **GPU crash or hang.** Use `--no-miopen`.
- **`--device cuda` but "PyTorch sees no GPU".** The CPU build of PyTorch is installed, or ROCm is not set up. Check with `python -c "import torch; print(torch.version.hip, torch.cuda.is_available())"`.
- **"Could not download/load" the model.** The model may be gated (`preslaff/omnivoice-bg` is): accept its conditions on the Hugging Face model page and run `hf auth login`.
- **"ffmpeg not found".** Install it, e.g. `sudo apt install ffmpeg`.
- **"PyTorch is not installed".** Install the CPU or ROCm build first, see [Installation](#installation).
- **Unsupported format.** Only `.epub` and `.txt` (also `.text`) are read. Convert other formats first, e.g. `ebook-convert book.mobi book.epub` (Calibre).
- **A voice description is ignored.** Voice design needs the base model; keep the default `--design-model k2-fsa/OmniVoice`.

## Project layout

```text
book2audio.py          wrapper so `python book2audio.py ...` keeps working
book2audio/
  __main__.py          main(): parses the options, runs Book2Audio, turns errors into exit code 1
  app.py               Book2Audio: wires the pieces below together from the options
  args.py              command-line options
  config.py            constants (models, port, sample rate, pauses)
  exceptions.py        Book2AudioException and subclasses
  logging.py           log / debug / error (plain print)
  workdir.py           WorkDir: owner of every file name in the work folder
  loaders/             Loader base, EpubLoader, TxtLoader, for_path() picks one by file type
  text/                Lexicon (pronunciation fixes), Chunker (chapter -> speech/pause steps)
  tts/                 Narrator (OmniVoice wrapper), RemoteNarrator (the same over HTTP, remote.py),
                       SampleVoice / DesignedVoice, VoiceLibrary
  server.py            book2audio-server: serves Narrators over HTTP for --server
  output/              ChapterRenderer, M4bBuilder, plain audio helpers
  utils.py             small helpers
tests/                 pytest suite
```

## Tests

The tests need neither the model nor network access:

```bash
pip install -e ".[test]"
pytest
```
