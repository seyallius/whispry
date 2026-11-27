//! A small file browser, shown over the setup screen.

use std::path::{Path, PathBuf};

use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Flex, Layout, Rect};
use ratatui::style::{Modifier, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Clear, Paragraph};

use super::theme::{self, Hit};

/// File types worth offering. FFmpeg can read more; type the path for those.
const MEDIA_EXTENSIONS: &[&str] = &[
    "mp3", "wav", "m4a", "flac", "ogg", "oga", "opus", "aac", "wma", "aiff", "amr", "mp4", "m4v",
    "mkv", "mov", "avi", "webm", "wmv", "flv", "3gp", "ts",
];

const PAGE: usize = 10;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EntryKind {
    Parent,
    Folder,
    File,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    pub name: String,
    pub path: PathBuf,
    pub kind: EntryKind,
}

#[derive(Debug, PartialEq, Eq)]
pub enum PickerAction {
    None,
    Cancel,
    Choose(PathBuf),
}

pub struct PickerState {
    /// First row shown; only moves when the highlight leaves the view.
    offset: std::cell::Cell<usize>,
    dir: PathBuf,
    entries: Vec<Entry>,
    selected: usize,
    error: Option<String>,
}

pub fn is_media(path: &Path) -> bool {
    path.extension()
        .and_then(|extension| extension.to_str())
        .is_some_and(|extension| MEDIA_EXTENSIONS.contains(&extension.to_lowercase().as_str()))
}

impl PickerState {
    pub fn open(dir: &Path) -> Self {
        let mut state = PickerState {
            offset: std::cell::Cell::new(0),
            dir: PathBuf::new(),
            entries: Vec::new(),
            selected: 0,
            error: None,
        };
        // An absolute path, so every parent folder is a real place to go to.
        let dir = std::path::absolute(dir).unwrap_or_else(|_| dir.to_path_buf());
        state.load(dir, None);
        state
    }

    #[cfg(test)]
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Show `dir`, selecting the entry at `keep` (a path) if it is there.
    fn load(&mut self, dir: PathBuf, keep: Option<&Path>) {
        let mut folders = Vec::new();
        let mut files = Vec::new();
        match std::fs::read_dir(&dir) {
            Ok(read) => {
                self.error = None;
                for item in read.flatten() {
                    let name = item.file_name().to_string_lossy().into_owned();
                    if name.starts_with('.') {
                        continue;
                    }
                    let path = item.path();
                    if path.is_dir() {
                        folders.push(Entry {
                            name,
                            path,
                            kind: EntryKind::Folder,
                        });
                    } else if is_media(&path) {
                        files.push(Entry {
                            name,
                            path,
                            kind: EntryKind::File,
                        });
                    }
                }
            }
            Err(error) => {
                // Stay where we are and say why.
                self.error = Some(format!("Cannot open {}: {error}", dir.display()));
                return;
            }
        }
        folders.sort_by_key(|entry| entry.name.to_lowercase());
        files.sort_by_key(|entry| entry.name.to_lowercase());

        self.entries.clear();
        if let Some(parent) = dir.parent() {
            self.entries.push(Entry {
                name: "..".into(),
                path: parent.to_path_buf(),
                kind: EntryKind::Parent,
            });
        }
        self.entries.extend(folders);
        self.entries.extend(files);
        self.offset.set(0);
        self.selected = keep
            .and_then(|keep| self.entries.iter().position(|entry| entry.path == keep))
            .unwrap_or(usize::from(
                self.entries.len() > 1 && dir.parent().is_some(),
            ));
        self.dir = dir;
    }

    fn go_up(&mut self) {
        if let Some(parent) = self.dir.parent().map(Path::to_path_buf) {
            let came_from = self.dir.clone();
            self.load(parent, Some(&came_from));
        }
    }

    fn select(&mut self, index: isize) {
        if !self.entries.is_empty() {
            self.selected = index.clamp(0, self.entries.len() as isize - 1) as usize;
        }
    }

    /// Jump to the next entry whose name starts with `letter`.
    fn jump_to(&mut self, letter: char) {
        let letter = letter.to_lowercase().next().unwrap_or(letter);
        let count = self.entries.len();
        for offset in 1..=count {
            let index = (self.selected + offset) % count;
            let entry = &self.entries[index];
            if entry.kind != EntryKind::Parent && entry.name.to_lowercase().starts_with(letter) {
                self.selected = index;
                return;
            }
        }
    }

    /// Move the highlight by `delta` rows (the mouse wheel).
    pub fn scroll(&mut self, delta: isize) {
        self.select(self.selected as isize + delta);
    }

    /// Show the `index`th folder of the current path (0 is the root).
    pub fn go_to_ancestor(&mut self, index: usize) {
        let mut chain: Vec<PathBuf> = self.dir.ancestors().map(Path::to_path_buf).collect();
        chain.reverse();
        if let Some(target) = chain.get(index).cloned() {
            let came_from = self.dir.clone();
            self.load(target, Some(&came_from));
        }
    }

    pub fn is_selected(&self, index: usize) -> bool {
        self.selected == index
    }

    /// A click on row `index`: the first click selects, a click on the
    /// highlighted row opens it, like Enter.
    pub fn click(&mut self, index: usize) -> PickerAction {
        if index >= self.entries.len() {
            return PickerAction::None;
        }
        if index == self.selected {
            return self.handle_key(&KeyEvent::from(KeyCode::Enter));
        }
        self.selected = index;
        PickerAction::None
    }

    pub fn handle_key(&mut self, key: &KeyEvent) -> PickerAction {
        let selected = self.selected as isize;
        if key.modifiers.contains(KeyModifiers::CONTROL) {
            return match key.code {
                KeyCode::Char('c' | 'q') => PickerAction::Cancel,
                _ => PickerAction::None,
            };
        }
        match key.code {
            KeyCode::Esc => return PickerAction::Cancel,
            KeyCode::Up => self.select(selected - 1),
            KeyCode::Down => self.select(selected + 1),
            KeyCode::PageUp => self.select(selected - PAGE as isize),
            KeyCode::PageDown => self.select(selected + PAGE as isize),
            KeyCode::Home => self.select(0),
            KeyCode::End => self.select(isize::MAX / 2),
            KeyCode::Backspace | KeyCode::Left => self.go_up(),
            KeyCode::Enter | KeyCode::Right => {
                if let Some(entry) = self.entries.get(self.selected).cloned() {
                    match entry.kind {
                        EntryKind::Parent => self.go_up(),
                        EntryKind::Folder => self.load(entry.path, None),
                        EntryKind::File => return PickerAction::Choose(entry.path),
                    }
                }
            }
            KeyCode::Char('~') => {
                if let Some(home) = std::env::home_dir() {
                    self.load(home, None);
                }
            }
            KeyCode::Char('/') => {
                if let Some(root) = self.dir.ancestors().last().map(Path::to_path_buf) {
                    self.load(root, None);
                }
            }
            KeyCode::Char(letter) => self.jump_to(letter),
            _ => {}
        }
        PickerAction::None
    }

    pub fn render(&self, frame: &mut Frame, area: Rect) {
        theme::add(area, Hit::PickerOutside);
        let [popup] = Layout::horizontal([Constraint::Percentage(80)])
            .flex(Flex::Center)
            .areas(area);
        let [popup] = Layout::vertical([Constraint::Percentage(85)])
            .flex(Flex::Center)
            .areas(popup);
        frame.render_widget(Clear, popup);
        theme::add(popup, Hit::Nothing);

        let block = Block::bordered()
            .border_type(theme::BORDER)
            .border_style(Style::new().fg(theme::accent()))
            .title(Span::styled(
                " Choose a file ",
                Style::new()
                    .fg(theme::accent())
                    .add_modifier(Modifier::BOLD),
            ));
        let inner = block.inner(popup);
        frame.render_widget(block, popup);

        let [path_area, list_area, footer_area] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Min(1),
            Constraint::Length(1),
        ])
        .areas(inner);

        // The path, each folder in it clickable.
        let mut chain: Vec<&Path> = self.dir.ancestors().collect();
        chain.reverse();
        let mut start = 0;
        for (index, folder) in chain.iter().enumerate() {
            let end = folder.display().to_string().chars().count();
            let x = (path_area.x + start as u16).min(path_area.right());
            let width = ((end - start) as u16).min(path_area.right() - x);
            let rect = Rect::new(x, path_area.y, width, 1);
            let style = if theme::hovered(rect) {
                Style::new()
                    .fg(theme::accent2())
                    .add_modifier(Modifier::UNDERLINED)
            } else {
                theme::dim_style()
            };
            let text: String = self
                .dir
                .display()
                .to_string()
                .chars()
                .skip(start)
                .take(end - start)
                .collect();
            frame.render_widget(Paragraph::new(Span::styled(text, style)), rect);
            theme::add(rect, Hit::PickerCrumb(index));
            start = end;
        }

        if self.entries.is_empty() {
            frame.render_widget(
                Paragraph::new(Span::styled(
                    "No audio or video files here",
                    theme::dim_style(),
                )),
                list_area,
            );
        }
        // Scroll just far enough to keep the highlighted row in view.
        let height = list_area.height as usize;
        let mut first = self.offset.get();
        if self.selected < first {
            first = self.selected;
        } else if self.selected >= first + height {
            first = self.selected + 1 - height;
        }
        self.offset.set(first);
        for (offset, (index, entry)) in self
            .entries
            .iter()
            .enumerate()
            .skip(first)
            .take(height)
            .enumerate()
        {
            let rect = Rect::new(list_area.x, list_area.y + offset as u16, list_area.width, 1);
            let (icon, mut style) = match entry.kind {
                EntryKind::Parent => ("↰ ", theme::dim_style()),
                EntryKind::Folder => ("▸ ", Style::new().fg(theme::accent())),
                EntryKind::File => ("♪ ", Style::new().fg(theme::accent2())),
            };
            if index == self.selected {
                style = style.bg(theme::surface()).add_modifier(Modifier::BOLD);
            } else if theme::hovered(rect) {
                style = style.add_modifier(Modifier::UNDERLINED);
            }
            let line = format!(" {icon}{}", entry.name);
            let pad = (rect.width as usize).saturating_sub(line.chars().count());
            frame.render_widget(
                Paragraph::new(Span::styled(format!("{line}{}", " ".repeat(pad)), style)),
                rect,
            );
            theme::add(rect, Hit::PickerRow(index));
        }

        let hint = match &self.error {
            Some(error) => Span::styled(error.clone(), Style::new().fg(theme::error())),
            None => Span::styled(
                "Enter open · Backspace up · ~ home · / root · click the path · Esc cancel",
                theme::dim_style(),
            ),
        };
        frame.render_widget(Paragraph::new(hint), footer_area);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    /// talks/{old/, b.mp3, A.WAV, notes.txt, .hidden.mp3}
    fn tree() -> (tempfile::TempDir, PathBuf) {
        let dir = tempfile::tempdir().unwrap();
        let talks = dir.path().join("talks");
        std::fs::create_dir_all(talks.join("old")).unwrap();
        for name in ["b.mp3", "A.WAV", "notes.txt", ".hidden.mp3"] {
            std::fs::write(talks.join(name), b"").unwrap();
        }
        (dir, talks)
    }

    fn names(state: &PickerState) -> Vec<&str> {
        state.entries().iter().map(|e| e.name.as_str()).collect()
    }

    #[test]
    fn lists_folders_first_then_media_files_without_hidden_or_other_files() {
        let (_dir, talks) = tree();
        let state = PickerState::open(&talks);
        assert_eq!(names(&state), ["..", "old", "A.WAV", "b.mp3"]);
    }

    #[test]
    fn the_first_real_entry_is_selected_not_the_parent() {
        let (_dir, talks) = tree();
        let mut state = PickerState::open(&talks);
        assert_eq!(state.handle_key(&key(KeyCode::Enter)), PickerAction::None);
        // It opened the "old" folder, which is empty.
        assert_eq!(names(&state), [".."]);
    }

    #[test]
    fn going_up_selects_the_folder_we_came_from() {
        let (_dir, talks) = tree();
        let mut state = PickerState::open(&talks);
        state.handle_key(&key(KeyCode::Enter)); // into old
        state.handle_key(&key(KeyCode::Backspace)); // back to talks
        assert_eq!(state.entries()[state.selected].name, "old");
        state.handle_key(&key(KeyCode::Backspace)); // up from talks
        assert_eq!(state.entries()[state.selected].name, "talks");
    }

    #[test]
    fn choosing_a_file_returns_its_path() {
        let (_dir, talks) = tree();
        let mut state = PickerState::open(&talks);
        state.handle_key(&key(KeyCode::Down)); // A.WAV
        assert_eq!(
            state.handle_key(&key(KeyCode::Enter)),
            PickerAction::Choose(talks.join("A.WAV"))
        );
    }

    #[test]
    fn typing_a_letter_jumps_to_a_matching_name() {
        let (_dir, talks) = tree();
        let mut state = PickerState::open(&talks);
        state.handle_key(&key(KeyCode::Char('b')));
        assert_eq!(state.entries()[state.selected].name, "b.mp3");
        state.handle_key(&key(KeyCode::Char('a')));
        assert_eq!(state.entries()[state.selected].name, "A.WAV");
    }

    #[test]
    fn the_selection_stays_inside_the_list_and_escape_cancels() {
        let (_dir, talks) = tree();
        let mut state = PickerState::open(&talks);
        state.handle_key(&key(KeyCode::End));
        assert_eq!(state.selected, 3);
        state.handle_key(&key(KeyCode::Down));
        assert_eq!(state.selected, 3);
        state.handle_key(&key(KeyCode::Home));
        assert_eq!(state.selected, 0);
        state.handle_key(&key(KeyCode::Up));
        assert_eq!(state.selected, 0);
        assert_eq!(state.handle_key(&key(KeyCode::Esc)), PickerAction::Cancel);
    }

    #[test]
    fn an_unreadable_folder_keeps_the_view_and_shows_the_reason() {
        let (_dir, talks) = tree();
        let mut state = PickerState::open(&talks);
        state.load(talks.join("missing"), None);
        assert!(state.error.as_deref().unwrap().contains("Cannot open"));
        assert_eq!(names(&state), ["..", "old", "A.WAV", "b.mp3"]);
    }

    #[test]
    fn a_relative_folder_is_made_absolute_so_it_can_go_up() {
        let state = PickerState::open(Path::new("."));
        assert!(state.dir.is_absolute());
        assert_eq!(state.entries()[0].kind, EntryKind::Parent);
    }

    #[test]
    fn recognises_media_by_extension_ignoring_case() {
        assert!(is_media(Path::new("a.MP3")));
        assert!(is_media(Path::new("/x/y.mkv")));
        assert!(!is_media(Path::new("a.txt")));
        assert!(!is_media(Path::new("mp3")));
    }
}
