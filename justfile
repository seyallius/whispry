# Whispry developer recipes. Users don't need just: see the README.

python := if os_family() == "windows" { "python" } else { "python3" }

default:
    @just --list

# Build the program (pass --release for an optimized build).
build *flags:
    cargo build {{flags}}

# Run Whispry from the checkout (no arguments opens the terminal UI).
run *args:
    cargo run -q -- {{args}}

# Run every test: the Rust crates and the packaging scripts.
test:
    cargo test --workspace
    {{python}} -m pytest -q scripts

# Check formatting and lints.
lint:
    cargo fmt --all --check
    cargo clippy --workspace --all-targets -- -D warnings

# Show the detected OS, devices (CPU/GPU) and FFmpeg.
check:
    cargo run -q -- devices

# Download and cache a Whisper model.
download model="small":
    cargo run -q -- download {{model}}

# Build a release archive for this machine (version from Cargo.toml).
package target:
    cargo build --release
    {{python}} scripts/package.py --binary target/release/whispry --version "$(grep -m1 '^version' Cargo.toml | cut -d'"' -f2)" --target {{target}}

# Remove build output.
clean:
    cargo clean
