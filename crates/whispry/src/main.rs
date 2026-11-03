mod cli;
mod console;

use clap::Parser;

fn main() -> std::process::ExitCode {
    cli::run(cli::Cli::parse())
}
