<h1 align="center">Whispry</h1>

<p align="center">
  <b>Turn recordings into text, right on your own computer.</b><br>
  No uploads, no accounts, no subscriptions. Just your audio and your machine.
</p>

<p align="center">
  <img src="docs/screenshots/transcribing.png" alt="Whispry transcribing a meeting recording, with a live transcript and progress" width="820">
</p>

Whispry is a friendly terminal app for speech-to-text. Point it at a
meeting, lecture, interview, or voice memo, and watch the transcript
appear line by line. It runs on Linux, Windows, and macOS, uses your
NVIDIA GPU when you have one, and works fine on a plain laptop CPU when
you don't.

Under the hood it uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and CTranslate2, OpenAI's Whisper models made fast.

## Quick start

```bash
git clone https://github.com/seyallius/whispry.git
cd whispry
python3 scripts/setup.py   # Windows: py scripts\setup.py
.venv/bin/whispry          # Windows: .venv\Scripts\whispry
```

That's it. Setup figures out whether you have a GPU and whether FFmpeg
is installed, and asks you only when it needs a decision.

## A quick tour

**Pick what to transcribe.** Choose a file (type a path or browse),
a model, the language, and where to run it. Whispry checks your machine
first: if there's no usable GPU, that option is greyed out and tells you
why.

<p align="center">
  <img src="docs/screenshots/setup.png" alt="Setup screen with file, model, language and device options" width="720">
</p>

**Watch it work.** The transcript fills in as Whispry listens, with
progress, speed, and an estimate of the time left. Changed your mind?
Press `Esc`; anything already transcribed is kept.

**Get your text.** A timestamped `.txt` file lands right next to your
recording.

<p align="center">
  <img src="docs/screenshots/done.png" alt="Finished transcription with the saved file path" width="720">
</p>

**Missing FFmpeg? No problem.** Whispry tells you how to install it for
your system, or sets up a bundled copy for you. Your call.

<p align="center">
  <img src="docs/screenshots/ffmpeg.png" alt="Dialog offering install steps or a bundled FFmpeg" width="720">
</p>

_Screenshots show demo content._

## Features

- **Private by design:** your audio never leaves your computer
- **CPU or GPU:** runs anywhere and uses an NVIDIA GPU automatically when
  available, falling back to the CPU if the GPU has trouble
- **Handles long recordings:** audio is processed a few minutes at a time,
  so a three-hour recording uses no more memory than a three-minute one
- **Persian and English**, or let Whispry detect the language
- **Small and Medium models**, downloaded once and then used offline
- **Terminal UI for people, CLI for scripts**, both powered by the same engine
- **Timestamped UTF-8 text** you can search, edit, or share

## Requirements

- Python 3.10 or newer
- FFmpeg (Whispry can install a bundled copy for you, see below)
- Optional: an NVIDIA GPU with a recent driver (Linux or Windows)
- Internet access for installation and the first download of each model

## Installation

The [quick start](#quick-start) covers most people. In more detail:

Setup creates a `.venv` folder and installs Whispry into it. It installs
the CUDA libraries only when it detects an NVIDIA GPU. Pass `--gpu` or
`--cpu` to override.

### FFmpeg

If FFmpeg is not on your PATH, setup (and the app) asks you to choose:

1. **Install FFmpeg yourself.** Whispry shows the steps for your system
   (`winget install --id Gyan.FFmpeg -e` on Windows, `brew install ffmpeg`
   on macOS, your package manager on Linux). Choose this if other programs
   also use FFmpeg.
2. **Use the bundled FFmpeg.** This installs
   [imageio-ffmpeg](https://pypi.org/project/imageio-ffmpeg/) into
   Whispry's own environment. It is a full FFmpeg build, about 30–40 MB to
   download and 70–170 MB on disk depending on the OS.

Nothing is installed unless you pick option 2. You can run the check again
at any time with `whispry ffmpeg`.

## Usage

Activate the environment (`source .venv/bin/activate`, or
`.venv\Scripts\activate` on Windows), then run:

```bash
whispry
```

This opens the terminal UI. Choose a file, model, language, and device,
then press **Start** (or `Ctrl+S`). The GPU option is greyed out with the
reason when it can't be used. During transcription, press `Esc` to cancel;
text transcribed so far is kept.

Without activating, run `.venv/bin/whispry` (Windows:
`.venv\Scripts\whispry`).

### Command line

```bash
whispry transcribe recording.mp3
whispry transcribe recording.mp3 --language fa --model medium
whispry transcribe recording.mp4 --device cpu --output notes.txt
whispry devices          # OS, CPU/GPU availability, FFmpeg in use
whispry download medium  # pre-download a model for offline use
```

`--device` accepts `auto` (default: GPU if available, else CPU), `cpu`,
or `cuda`. `--chunk-minutes` sets how much audio is processed at once
(default 10; lower uses less RAM).

The transcription is written to a `.txt` file beside the input unless
`--output` is given.

## Models

Whispry uses the Small model by default. Each model is downloaded from
Hugging Face on first use and cached in:

- Linux: `~/.cache/whispry/models`
- Windows: `%LOCALAPPDATA%\whispry\models`
- macOS: `~/Library/Caches/whispry/models`

Set `WHISPRY_MODEL_DIR` to use another folder. Models downloaded by
earlier versions into the repository's `models/` folder are still used.

After the model is cached, transcription works offline.

## How it works

1. Detect the OS, whether a usable NVIDIA GPU is present, and FFmpeg.
2. Locate or download the selected model, then load it on the chosen
   device. On the GPU, CUDA libraries installed by pip are loaded from
   inside Python, with no environment variables or wrapper scripts.
3. Stream the decoded audio from FFmpeg in blocks.
4. Group blocks into windows of about 10 minutes, ending each window
   inside a pause (found with Silero VAD) so no word is cut in half.
5. Transcribe each window. The language is detected once and then kept,
   and each window gets the end of the previous one as context.
6. Write each timestamped line to the output file as soon as it is ready.

The code is split into two layers:

- `whispry/core/`: all logic (detection, FFmpeg, streaming, models,
  transcription). It never prints. It reports progress through typed
  events and can be cancelled from another thread.
- `whispry/tui/` and `whispry/cli.py`: front ends that only turn user
  input into a job and render the events it emits.

## Development

The `justfile` holds developer recipes. Run `just` to list them.

```bash
just setup   # create .venv with dev tools
just test    # run the test suite
just check   # show detected devices and FFmpeg
just run     # run Whispry from the checkout
```

## Limitations

- GPU acceleration requires an NVIDIA GPU with CUDA 12 support. Apple
  Silicon and AMD GPUs run on the CPU.
- The CPU is much slower than a GPU, especially with the Medium model.
- Progress is estimated from completed segments.
- The Small model may be less accurate for difficult Persian audio.

## Privacy

Audio processing happens locally.

Model downloads contact Hugging Face when a model is missing.
No audio file is intentionally uploaded by the application.

## License

See LICENSE.
