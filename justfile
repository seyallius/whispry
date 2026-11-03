# Whispry development and transcription recipes.

python := if os_family() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }

default:
    @just --list

# Install Python dependencies and CUDA runtime libraries.
setup:
    ./scripts/setup.sh

# Show the detected OS and which devices (CPU/GPU) can run Whisper.
check:
    {{python}} -c "from whispry.core.platform import detect_system; s = detect_system(); print(s.os_label); [print('-', d.label, 'available' if d.available else 'unavailable:', d.reason or ', '.join(d.compute_types)) for d in s.devices]"

# Transcribe with automatic language detection.
transcribe file model="small":
    {{python}} -m whispry transcribe "{{file}}" --model "{{model}}"

# Transcribe Persian speech.
persian file model="small":
    {{python}} -m whispry transcribe "{{file}}" --language fa --model "{{model}}"

# Transcribe English speech.
english file model="small":
    {{python}} -m whispry transcribe "{{file}}" --language en --model "{{model}}"

# Download and cache a Whisper model.
download model="small":
    {{python}} -c "from whispry.core.models import ensure_model; print(ensure_model('{{model}}'))"

# Check Python syntax without executing the transcriber.
lint:
    {{python}} -c "import ast, pathlib; [ast.parse(p.read_text()) for p in pathlib.Path('whispry').rglob('*.py')]"

# Remove Python cache files.
clean:
    find . -type d -name __pycache__ -not -path './.venv/*' -prune -exec rm -rf {} +
