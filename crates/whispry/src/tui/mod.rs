//! The terminal UI. It only turns key presses into a job and draws what the
//! job reports; the work itself happens in `whispry-core`.

mod app;
mod file_picker;
mod input;
mod services;
mod setup;

#[cfg(test)]
mod tests;

use std::io;
use std::path::PathBuf;
use std::time::Duration;

use ratatui::DefaultTerminal;
use ratatui::crossterm::event::{
    self, DisableBracketedPaste, EnableBracketedPaste, Event, KeyEventKind,
};
use ratatui::crossterm::execute;
use whispry_core::config::JobConfig;

use app::App;
use services::RealServices;

/// How often the screen is redrawn when nothing happens.
const TICK: Duration = Duration::from_millis(100);

/// Open the terminal UI. Returns the job the user chose to start, if any.
pub fn run(initial_file: Option<PathBuf>) -> io::Result<Option<JobConfig>> {
    let mut app = App::new(Box::new(RealServices::new()), initial_file);
    let mut terminal = ratatui::init();
    let _ = execute!(io::stdout(), EnableBracketedPaste);
    let result = event_loop(&mut terminal, &mut app);
    let _ = execute!(io::stdout(), DisableBracketedPaste);
    ratatui::restore();
    result.map(|()| app.into_outcome())
}

fn event_loop(terminal: &mut DefaultTerminal, app: &mut App) -> io::Result<()> {
    while !app.is_done() {
        terminal.draw(|frame| app.render(frame))?;
        if event::poll(TICK)? {
            match event::read()? {
                // Windows also reports key releases; only presses count.
                Event::Key(key) if key.kind == KeyEventKind::Press => app.on_key(key),
                Event::Paste(text) => app.on_paste(&text),
                _ => {}
            }
        }
    }
    Ok(())
}
