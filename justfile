# Whispry developer recipes: setup, dependencies, checks.
# Users don't need just: run `python3 scripts/setup.py`, then `whispry`.

python := if os_family() == "windows" { ".venv/Scripts/python.exe" } else { ".venv/bin/python" }
system_python := if os_family() == "windows" { "python" } else { "python3" }

default:
    @just --list

# Create .venv and install Whispry with dev tools (GPU support auto-detected; pass --gpu or --cpu to override).
setup *flags:
    {{system_python}} scripts/setup.py --dev {{flags}}

# Reinstall dependencies after changing pyproject.toml.
deps *extras="dev":
    {{python}} -m pip install -e ".[{{extras}}]"

# Install the bundled FFmpeg build into .venv.
deps-ffmpeg:
    {{python}} -m pip install -e ".[ffmpeg]"

# Show the detected OS, devices (CPU/GPU), and FFmpeg.
check:
    {{python}} -m whispry devices

# Download and cache a Whisper model.
download model="small":
    {{python}} -m whispry download {{model}}

# Run Whispry from the checkout (no args opens the TUI).
run *args:
    {{python}} -m whispry {{args}}

# Run the test suite.
test *args:
    {{python}} -m pytest {{args}}

# Byte-compile every module to catch syntax errors.
lint:
    {{python}} -m compileall -q whispry tests scripts

# Remove Python cache files.
clean:
    {{python}} -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]"

# --- Rust workspace ---

# Build the Rust binary (pass --release for an optimized build).
rs-build *flags:
    cargo build {{flags}}

# Run the Rust test suite.
rs-test *args:
    cargo test --workspace {{args}}

# Check formatting and lints.
rs-lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings
