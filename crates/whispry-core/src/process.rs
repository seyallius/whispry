//! Helpers for starting child processes the same way on every OS.

use std::process::Command;

/// A command that does not flash a console window on Windows, and does not
/// receive the terminal's Ctrl+C on Unix.
///
/// Ctrl+C goes to the whole foreground process group. Children get their own
/// group, so the program can stop them in an orderly way and report that the
/// job was interrupted, instead of seeing them crash.
pub(crate) fn command(program: impl AsRef<std::ffi::OsStr>) -> Command {
    #[allow(unused_mut)]
    let mut command = Command::new(program);
    #[cfg(unix)]
    {
        use std::os::unix::process::CommandExt;
        command.process_group(0);
    }
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        command.creation_flags(CREATE_NO_WINDOW);
    }
    command
}
