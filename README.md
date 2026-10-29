# Whispry

Private, local GPU-accelerated speech-to-text transcription.

Whispry uses [faster-whisper](https://github.com/SYSTRAN/faster-whisper)
and CTranslate2 to transcribe audio and video files on your own computer.

## Features

- Local transcription without uploading recordings to a cloud service
- NVIDIA CUDA GPU acceleration
- Persian and English transcription
- Automatic language detection
- Small and Medium Whisper models
- Visible model download and transcription progress
- Timestamped UTF-8 text output
- Automatic FFmpeg audio decoding
- Locally cached Whisper models
- Reusable Just recipes

## Prerequisites

- Linux (developed on Ubuntu 24.04)
- Python 3.12 or a compatible Python 3 version
- FFmpeg
- NVIDIA GPU with a compatible driver
- CUDA-compatible CTranslate2 runtime
- Git
- Just (optional, recommended)
- Internet access for initial installation and model downloads

The initial development machine uses an NVIDIA RTX 3050 Ti
with 4 GB VRAM.

Other GPUs and platforms have not yet been tested.

## Installation

Clone the repository:

```bash
git clone https://github.com/seyallius/whispry.git
cd whispry
````

Install the system prerequisites on Ubuntu:

```
sudo apt install python3-venv python3-pip ffmpeg
```

Install Just using a supported package manager.

Set up the Python environment:

```
just setup
```

Verify GPU access:

```
just check
```

The expected result includes at least one CUDA device.

## Usage

Transcribe an English recording:

```
just english "/path/to/recording.wav"
```

Transcribe Persian:

```
just persian "/path/to/recording.mp3"
```

Automatically detect the language:

```
just transcribe "/path/to/recording.mp4"
```

Use the Medium model:

```
just persian "/path/to/recording.wav" medium
```

Or invoke Python directly:

```
./scripts/run.sh transcribe.py "/path/to/audio.wav" --language fa
```

The transcription is written to a `.txt` file beside the input.

## Model downloads

Whispry uses the Small model by default.

The first run downloads the model if it is not already cached.

You can also download models manually:

```
just download small
just download medium
```

Models are stored under `models/`, which is excluded from Git.

After the model and dependencies are installed, transcription can run\
without an internet connection.

## How it works

1. Locate or download the selected Whisper model.
2. Load the model using CTranslate2 with CUDA acceleration.
3. Decode the input audio using FFmpeg.
4. Run speech recognition with faster-whisper.
5. Display approximate transcription progress.
6. Save timestamped text to a UTF-8 file.

FFmpeg decoding avoids a known incompatibility between the PyAV API\
and the faster-whisper version used during development.

## Limitations

- Currently optimized for NVIDIA CUDA on Linux.
- Audio decoding loads uncompressed samples into system RAM.
- Progress is estimated from completed audio segments.
- Model downloads require internet access on first use.
- The Small model may be less accurate for difficult Persian audio.
- This project is experimental and not yet packaged for distribution.

## Privacy

Audio processing happens locally.

Model downloads contact Hugging Face when a model is missing.\
No audio file is intentionally uploaded by the application.

## License

See LICENSE.\
