//! Turns core events into printed lines and progress bars.

use std::collections::HashMap;

use indicatif::{ProgressBar, ProgressDrawTarget, ProgressStyle};
use whispry_core::events::{Event, Stage};
use whispry_core::timestamps::format_timestamp;

/// The plain line an event prints, if it prints one.
pub fn describe(event: &Event) -> Option<String> {
    match event {
        Event::StageStarted { message, .. } => Some(format!("\n{message}")),
        Event::StageFinished { elapsed, .. } => Some(format!("Done in {elapsed:.1}s")),
        Event::LanguageDetected {
            language,
            probability,
        } => Some(format!(
            "Language: {language} ({:.0}%)",
            probability * 100.0
        )),
        Event::SegmentReady { start, text, .. } => {
            Some(format!("[{}] {text}", format_timestamp(*start)))
        }
        Event::Log(message) => Some(message.clone()),
        Event::Finished {
            output, elapsed, ..
        } => Some(format!(
            "\nCompleted in {elapsed:.1}s\nSaved transcription to: {}",
            output.display()
        )),
        Event::Progress { .. } => None,
    }
}

fn stage_name(stage: Stage) -> &'static str {
    match stage {
        Stage::Model => "Model",
        Stage::Transcribe => "Transcribe",
    }
}

fn bar_style(unit: &str) -> ProgressStyle {
    let template = if unit == "B" {
        "{prefix:>10} [{bar:30}] {bytes}/{total_bytes} {bytes_per_sec} eta {eta}"
    } else {
        "{prefix:>10} [{bar:30}] {msg} elapsed {elapsed}"
    };
    ProgressStyle::with_template(template)
        .expect("the template is valid")
        .progress_chars("=> ")
}

/// Prints events as they happen, with a progress bar for each stage.
pub struct ConsoleRenderer {
    bars: HashMap<&'static str, ProgressBar>,
}

impl ConsoleRenderer {
    pub fn new() -> Self {
        ConsoleRenderer {
            bars: HashMap::new(),
        }
    }

    /// Print a line without breaking a progress bar that is being drawn.
    fn print(&self, line: &str) {
        match self.bars.values().next() {
            Some(bar) => bar.println(line),
            None => println!("{line}"),
        }
    }

    fn update_bar(&mut self, stage: Stage, completed: f64, total: Option<f64>, unit: &str) {
        let name = stage_name(stage);
        let bar = self.bars.entry(name).or_insert_with(|| {
            let bar = match total {
                Some(total) => ProgressBar::new(total as u64),
                None => ProgressBar::new_spinner(),
            };
            bar.set_draw_target(ProgressDrawTarget::stderr_with_hz(10));
            bar.set_style(bar_style(unit));
            bar.set_prefix(name);
            bar
        });
        bar.set_position(completed as u64);
        if unit != "B" {
            let done = format_timestamp(completed);
            bar.set_message(match total {
                Some(total) => format!("{done} / {}", format_timestamp(total)),
                None => done,
            });
        }
    }

    pub fn handle(&mut self, event: &Event) {
        if let Event::Progress {
            stage,
            completed,
            total,
            unit,
        } = event
        {
            self.update_bar(*stage, *completed, *total, unit);
            return;
        }
        if let Event::StageFinished { stage, .. } = event
            && let Some(bar) = self.bars.remove(stage_name(*stage))
        {
            bar.finish_and_clear();
        }
        if let Some(line) = describe(event) {
            self.print(&line);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    #[test]
    fn describes_the_events_that_print_a_line() {
        let described = |event| describe(&event);
        assert_eq!(
            described(Event::LanguageDetected {
                language: "fa".into(),
                probability: 0.915
            }),
            Some("Language: fa (92%)".into())
        );
        assert_eq!(
            described(Event::SegmentReady {
                start: 65.0,
                end: 70.0,
                text: "hello".into()
            }),
            Some("[01:05] hello".into())
        );
        assert_eq!(
            described(Event::StageFinished {
                stage: Stage::Model,
                elapsed: 2.34
            }),
            Some("Done in 2.3s".into())
        );
        assert_eq!(
            described(Event::Finished {
                output: PathBuf::from("out.txt"),
                audio_duration: 10.0,
                elapsed: 5.0
            }),
            Some("\nCompleted in 5.0s\nSaved transcription to: out.txt".into())
        );
        assert_eq!(
            described(Event::Progress {
                stage: Stage::Model,
                completed: 1.0,
                total: None,
                unit: "B".into()
            }),
            None
        );
    }
}
