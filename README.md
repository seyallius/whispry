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
appear line by line. It runs on Linux, Windows and macOS, works on a plain
laptop CPU, and can use an NVIDIA GPU if you choose to set that up.

Under the hood it runs OpenAI's Whisper models with
[CTranslate2](https://github.com/OpenNMT/CTranslate2), built into the program
itself. It is a single Rust program: no Python, no virtual environment, no
setup step.

## Quick start

1. Install [FFmpeg](#ffmpeg) if you don't have it.
2. Download the archive for your system from the
   [Releases page](https://github.com/seyallius/whispry/releases) and unpack it.
3. Run it:

   ```bash
   cd whispry-0.3.0-linux-x86_64
   ./whispry                 # Windows: .\whispry.exe
   ```

That's it. The first time you transcribe, Whispry downloads the speech model
you picked (once; after that it works offline). The plain archives run on the
CPU and need no GPU. See [GPU acceleration](#gpu-acceleration-optional) if you
have an NVIDIA GPU.

## A quick tour

**Pick what to transcribe.** Choose a file (type a path, paste one, or
browse), a model, the language, and where to run it. Whispry checks your
machine first: if there's no usable GPU, that option is greyed out and tells
you why.

<p align="center">
  <img src="docs/screenshots/setup.png" alt="Setup screen with file, model, language and device options" width="720">
</p>

**Browse for a file.** Press `Ctrl+O` to look around. Folders come first,
then your audio and video files.

<p align="center">
  <img src="docs/screenshots/browser.png" alt="File browser listing folders and recordings" width="720">
</p>

**Watch it work.** The transcript fills in as Whispry listens, with
progress, speed, and an estimate of the time left. Changed your mind?
Press `Esc`; anything already transcribed is kept.

**Get your text.** A timestamped `.txt` file lands right next to your
recording.

<p align="center">
  <img src="docs/screenshots/done.png" alt="Finished transcription with the saved file path" width="720">
</p>

**Missing FFmpeg? Whispry tells you what to do.** It shows the install
steps for your system and picks up where you left off once it's installed.

<p align="center">
  <img src="docs/screenshots/ffmpeg.png" alt="Dialog explaining how to install FFmpeg" width="720">
</p>

_Screenshots show demo content._

## Features

- **Private by design:** your audio never leaves your computer
- **CPU everywhere, GPU if you want it:** the standard edition runs on the
  CPU; an optional CUDA edition can use an NVIDIA GPU
- **Handles long recordings:** audio is processed a few minutes at a time,
  so a three-hour recording needs no more memory for audio than a
  three-minute one
- **Cuts at pauses, not mid-word:** each chunk ends in a quiet moment
- **Persian and English**, or let Whispry detect the language
- **Small and Medium models**, downloaded once and then used offline
- **Terminal UI for people, command line for scripts**, both powered by the
  same engine
- **Timestamped UTF-8 text** you can search, edit, or share

## Requirements

- Linux or Windows (64-bit Intel/AMD), or macOS on Apple silicon or Intel
- FFmpeg (see [below](#ffmpeg))
- Internet access for the first download of each model
- Optional, Linux and Windows only: an NVIDIA GPU with a recent driver, and
  the CUDA edition (see [GPU acceleration](#gpu-acceleration-optional))

## Installation

Download the archive for your system from the
[Releases page](https://github.com/seyallius/whispry/releases), unpack it, and
run `whispry`. There is nothing else to install. Archives are named
`whispry-<version>-<system>`:

| Archive | For |
|---|---|
| `linux-x86_64` | Linux, 64-bit Intel/AMD (built against a fairly old glibc; very old distributions may not work) |
| `windows-x86_64` | Windows, 64-bit |
| `macos-arm64` | macOS on Apple silicon (CPU only) |
| `macos-x86_64` | macOS on Intel (CPU only) |
| `linux-x86_64-cuda`, `windows-x86_64-cuda` | The same, with optional NVIDIA GPU support |

The CUDA editions may be missing from a release if their build failed; the
CPU editions are always published.

### GPU acceleration (optional)

The CPU edition works everywhere and is the right choice unless you have an
NVIDIA GPU. For a GPU:

1. Download the archive whose name ends in `-cuda`.
2. Run `whispry gpu` to see the GPU status.
3. Run `whispry gpu --install` to download the NVIDIA cuBLAS library
   (about 550 MB).

Nothing is downloaded unless you ask. If your computer has no NVIDIA GPU, no
CUDA downloads are offered at all. CUDA libraries that are already on your
computer are reused instead of downloaded again. Without them, the CUDA
edition simply runs on the CPU.

### Building from source

You need [Rust](https://rustup.rs), [CMake](https://cmake.org) and a C++
compiler (GCC or Clang on Linux, Xcode command line tools on macOS, Visual
Studio Build Tools on Windows). The first build compiles CTranslate2 from
source and takes several minutes.

```bash
git clone https://github.com/seyallius/whispry.git
cd whispry
cargo build --release
./target/release/whispry
```

For the CUDA edition, install the NVIDIA CUDA toolkit (12.x) and build with:

```bash
cargo build --release --features cuda
```

### FFmpeg

Whispry uses FFmpeg to read audio and video files. If it isn't installed,
the app shows these steps for your system, and `whispry ffmpeg` prints them
too:

- **Windows:** `winget install --id Gyan.FFmpeg -e`
- **Ubuntu or Debian:** `sudo apt install ffmpeg`
- **Fedora:** `sudo dnf install ffmpeg-free`
- **Arch:** `sudo pacman -S ffmpeg`

Open a new terminal afterwards so the change to `PATH` is picked up.

## Usage

Run `whispry` with no arguments to open the terminal UI. Choose a file,
model, language, and device, then press **Start**.

| Key | What it does |
|---|---|
| `Tab` / `Shift+Tab`, `↑` / `↓` | Move between fields |
| `←` / `→` | Change a choice |
| `Ctrl+O` | Browse for a file |
| `Ctrl+S` | Start transcribing |
| `Esc` | Cancel a running job (text so far is kept) |
| `Ctrl+Q` | Quit (cancels a running job first) |

You can also paste a path, quotes and all, straight into the file field.

### Command line

```bash
whispry transcribe recording.mp3
whispry transcribe recording.mp3 --language fa --model medium
whispry transcribe recording.mp4 --device cpu --output notes.txt
whispry devices          # OS, CPU and GPU, and FFmpeg
whispry gpu              # NVIDIA GPU status (CUDA edition)
whispry gpu --install    # download the cuBLAS library (~550 MB)
whispry ffmpeg           # check for FFmpeg, or show how to install it
whispry download medium  # pre-download a model for offline use
```

`--device` accepts `auto` (default: GPU if available, else CPU), `cpu`, or
`cuda`. `--chunk-minutes` sets how much audio is processed at once (default
10; lower uses less RAM). Press `Ctrl+C` to stop; the text so far stays in
the output file and Whispry exits with code 130.

The transcription is written to a `.txt` file beside the input unless
`--output` is given.

## Models

Whispry uses the Small model by default. Each model is downloaded from
Hugging Face on first use and cached in:

- Linux: `~/.cache/whispry/models`
- macOS: `~/Library/Caches/whispry/models`
- Windows: `%LOCALAPPDATA%\whispry\models`

Set `WHISPRY_MODEL_DIR` to use another folder. After a model is cached,
transcription works offline.

## How it works

1. Detect the OS, whether a usable NVIDIA GPU is present, and FFmpeg.
2. Find the selected model, or download it, and load it on the chosen device.
3. Stream the decoded audio from FFmpeg in small blocks.
4. Group blocks into windows of about 10 minutes, ending each window in the
   longest pause near its end, so no word is cut in half.
5. Send each window to the speech engine in pieces of at most 30 seconds,
   cut at pauses. The language is detected once and then kept.
6. Write each timestamped line to the output file as soon as it is ready.

The code is split into two parts:

- `crates/whispry-core/`: all the logic (detection, FFmpeg, streaming,
  chunking, models, the job runner). It never prints; it reports progress
  through typed events and can be cancelled from another thread.
- `crates/whispry/`: the terminal UI and the command line. They only turn
  user input into a job and draw the events it emits.

## Development

You need [Rust](https://rustup.rs), CMake and a C++ compiler (see
[Building from source](#building-from-source)). Python 3 with `pytest` is
needed only for the packaging script tests, and
[just](https://github.com/casey/just) for the recipes. Run `just` to list them.

```bash
just test    # Rust tests and packaging script tests
just lint    # formatting and clippy
just run     # run Whispry from the checkout
just check   # show detected devices and FFmpeg
```

Tests that need FFmpeg are skipped when it is missing. Set
`WHISPRY_TEST_STRICT=1` to make them fail instead, as CI does.

To publish a release, push a tag such as `v0.3.0` (it must match the
version in `Cargo.toml`). GitHub Actions builds the archives for Linux,
Windows and macOS (plus the optional CUDA editions) and attaches them to a
new release.

## Limitations

- GPU acceleration requires an NVIDIA GPU, the CUDA edition, and Linux or
  Windows. Apple silicon GPUs and AMD GPUs are not supported; macOS runs on
  the CPU.
- The CPU is slower than a GPU, especially with the Medium model.
- Progress is estimated from completed segments.
- The Small model may be less accurate for difficult Persian audio.

## Privacy

Audio processing happens locally.

Model downloads contact Hugging Face when a model is missing, and
`whispry gpu --install` downloads the cuBLAS library when you ask for it. No
audio file is uploaded by the application.

## License

See LICENSE.
