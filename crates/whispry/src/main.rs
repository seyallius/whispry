mod cli;
mod console;
mod gpu;
mod tui;

use clap::Parser;

fn main() -> std::process::ExitCode {
    if let Some(code) = whispry_core::cuda_libs::relaunch_if_needed() {
        return std::process::ExitCode::from(code.clamp(0, 255) as u8);
    }
    cli::run(cli::Cli::parse())
}
