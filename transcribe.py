
#!/usr/bin/env python3
"""Local GPU-accelerated audio transcription with progress reporting."""

import argparse
import sys
import time
import subprocess
import numpy as np
import numpy as np
from pathlib import Path
from faster_whisper import WhisperModel
from huggingface_hub import snapshot_download
from tqdm import tqdm
from rich.console import Console


# -------------------------------------------- Constants --------------------------------------------

MODEL_REPOSITORIES = {
    "small": "Systran/faster-whisper-small",
    "medium": "Systran/faster-whisper-medium",
}

MODEL_FILES = [
    "config.json",
    "model.bin",
    "tokenizer.json",
    "vocabulary.*",
    "preprocessor_config.json",
]

MODEL_DIRECTORY = Path(__file__).resolve().parent / "models"
COMPUTE_TYPE = "int8_float16"
BEAM_SIZE = 5
SAMPLE_RATE = 16_000
BYTES_PER_SAMPLE = 4
CHUNK_SIZE = 1024 * 1024
console = Console()

def main():
    parser = argparse.ArgumentParser(
        description="Transcribe audio locally using NVIDIA GPU acceleration."
    )
    parser.add_argument("audio", type=Path, help="Audio or video file")
    parser.add_argument(
        "--language",
        choices=["fa", "en"],
        default=None,
        help="Language code; omit to auto-detect",
    )
    parser.add_argument(
        "--model",
        choices=MODEL_REPOSITORIES.keys(),
        default="small",
        help="Whisper model size",
    )

    args = parser.parse_args()

    if not args.audio.is_file():
        parser.error(f"File not found: {args.audio}")

    try:
        transcribe(args.audio, args.model, args.language)
    except KeyboardInterrupt:
        print("\nInterrupted by user.", file=sys.stderr)
        sys.exit(130)
    except Exception as error:
        # print(f"\nError: {error}", file=sys.stderr)
        # sys.exit(1)
        raise


def transcribe(audio: Path, model_name: str, language: str | None):
    model_path = download_model(model_name)

    print("\n[1/3] Loading model onto NVIDIA GPU...", flush=True)

    start = time.perf_counter()

    model = WhisperModel(
        str(model_path),
        device="cuda",
        compute_type=COMPUTE_TYPE,
        local_files_only=True,
    )

    print(f"Model loaded in {time.perf_counter() - start:.1f}s")

    print("\n[2/3] Preparing transcription...", flush=True)

    print("Decoding audio with FFmpeg...", flush=True)
    samples = decode_audio(audio)
    
    with console.status(
        "[cyan]Preparing audio (CPU voice detection and feature extraction)...",
        spinner="dots",
    ):
        segments, info = model.transcribe(
            samples,
            language=language,
            beam_size=BEAM_SIZE,
            vad_filter=True,
        )

    print(f"Language: {info.language}")
    print(f"Audio duration: {info.duration:.1f}s")

    output = audio.with_suffix(".txt")

    print("\n[3/3] Transcribing...", flush=True)

    start = time.perf_counter()
    last_position = 0.0

    with output.open("w", encoding="utf-8") as file:
        with tqdm(
            total=info.duration,
            desc="Transcribing",
            unit="s",
            dynamic_ncols=True,
        ) as progress:
            for segment in segments:
                current_position = min(segment.end, info.duration)
                delta = max(0.0, current_position - last_position)

                progress.update(delta)
                last_position = max(last_position, current_position)

                line = (
                    f"[{format_timestamp(segment.start)}] "
                    f"{segment.text.strip()}"
                )

                tqdm.write(line)
                file.write(line + "\n")
                file.flush()

            # Remaining duration may contain silence.
            progress.update(max(0.0, info.duration - last_position))

    elapsed = time.perf_counter() - start

    print(f"\nCompleted in {elapsed:.1f}s")
    print(f"Saved transcription to: {output}")

def download_model(model_name: str) -> Path:
    """Use a local model or download it if missing."""
    repository = MODEL_REPOSITORIES[model_name]
    destination = MODEL_DIRECTORY / model_name

    required_files = (
        "config.json",
        "model.bin",
        "tokenizer.json",
    )

    if all((destination / name).is_file() for name in required_files):
        print(f"Using cached model: {destination}", flush=True)
        return destination

    print(f"Downloading model: {model_name}", flush=True)

    snapshot_download(
        repo_id=repository,
        local_dir=str(destination),
        allow_patterns=MODEL_FILES,
    )

    return destination


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"
 
def get_audio_duration(audio: Path) -> float:
    """Return the media duration in seconds using FFprobe."""
    result = subprocess.run(
        [
            "ffprobe",
            "-v", "error",
            "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1",
            str(audio),
        ],
        capture_output=True,
        text=True,
        check=True,
    )

    return float(result.stdout.strip())


def decode_audio(audio: Path) -> np.ndarray:
    """Decode audio incrementally with visible progress."""
    duration = get_audio_duration(audio)

    command = [
        "ffmpeg",
        "-nostdin",
        "-v", "error",
        "-i", str(audio),
        "-f", "f32le",
        "-ac", "1",
        "-ar", str(SAMPLE_RATE),
        "pipe:1",
    ]

    process = subprocess.Popen(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    chunks = []
    decoded_bytes = 0
    expected_bytes = int(duration * SAMPLE_RATE * BYTES_PER_SAMPLE)

    try:
        with tqdm(
            total=expected_bytes,
            desc="Decoding audio",
            unit="B",
            unit_scale=True,
            dynamic_ncols=True,
        ) as progress:
            while True:
                chunk = process.stdout.read(CHUNK_SIZE)

                if not chunk:
                    break

                chunks.append(chunk)
                decoded_bytes += len(chunk)
                progress.update(len(chunk))

        return_code = process.wait()

        if return_code != 0:
            raise RuntimeError(
                f"FFmpeg decoding failed with exit code {return_code}"
            )

        return np.frombuffer(b"".join(chunks), dtype=np.float32)

    except BaseException:
        process.kill()
        process.wait()
        raise

    finally:
        if process.stdout:
            process.stdout.close()
    

if __name__ == "__main__":
    main()
