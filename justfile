# Whispry development and transcription recipes.

default:
    @just --list

# Install Python dependencies and CUDA runtime libraries.
setup:
    ./scripts/setup.sh

# Verify that CTranslate2 can access the NVIDIA GPU.
check:
    ./scripts/run.sh -c "import ctranslate2; print('CUDA devices:', ctranslate2.get_cuda_device_count()); print('Compute types:', ctranslate2.get_supported_compute_types('cuda'))"

# Transcribe with automatic language detection.
transcribe file model="small":
    ./scripts/run.sh transcribe.py "{{file}}" --model "{{model}}"

# Transcribe Persian speech.
persian file model="small":
    ./scripts/run.sh transcribe.py "{{file}}" --language fa --model "{{model}}"

# Transcribe English speech.
english file model="small":
    ./scripts/run.sh transcribe.py "{{file}}" --language en --model "{{model}}"

# Download and cache a Whisper model.
download model="small":
    hf download "Systran/faster-whisper-{{model}}" --local-dir "models/{{model}}"

# Check Python syntax without executing the transcriber.
lint:
    ./scripts/run.sh -c "import ast, pathlib; ast.parse(pathlib.Path('transcribe.py').read_text())"

# Remove Python cache files.
clean:
    find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
