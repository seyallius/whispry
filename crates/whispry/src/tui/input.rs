//! A single-line text field.

use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct TextInput {
    text: String,
    /// Position of the cursor, counted in characters.
    cursor: usize,
}

impl TextInput {
    pub fn new(text: &str) -> Self {
        TextInput {
            text: text.to_string(),
            cursor: text.chars().count(),
        }
    }

    pub fn text(&self) -> &str {
        &self.text
    }

    #[cfg(test)]
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    pub fn set(&mut self, text: &str) {
        *self = TextInput::new(text);
    }

    fn byte_index(&self, chars: usize) -> usize {
        self.text
            .char_indices()
            .nth(chars)
            .map_or(self.text.len(), |(index, _)| index)
    }

    /// Insert pasted or typed text at the cursor. Line breaks are dropped.
    pub fn insert(&mut self, text: &str) {
        let clean: String = text.chars().filter(|c| !c.is_control()).collect();
        let at = self.byte_index(self.cursor);
        self.text.insert_str(at, &clean);
        self.cursor += clean.chars().count();
    }

    /// Handle an editing key. Returns whether the key was used.
    pub fn handle_key(&mut self, key: &KeyEvent) -> bool {
        let plain = !key.modifiers.contains(KeyModifiers::CONTROL)
            && !key.modifiers.contains(KeyModifiers::ALT);
        match key.code {
            KeyCode::Char(c) if plain => self.insert(&c.to_string()),
            KeyCode::Backspace if self.cursor > 0 => {
                self.cursor -= 1;
                let at = self.byte_index(self.cursor);
                self.text.remove(at);
            }
            KeyCode::Delete if self.cursor < self.text.chars().count() => {
                let at = self.byte_index(self.cursor);
                self.text.remove(at);
            }
            KeyCode::Left => self.cursor = self.cursor.saturating_sub(1),
            KeyCode::Right => self.cursor = (self.cursor + 1).min(self.text.chars().count()),
            KeyCode::Home => self.cursor = 0,
            KeyCode::End => self.cursor = self.text.chars().count(),
            KeyCode::Char('u') if key.modifiers.contains(KeyModifiers::CONTROL) => {
                self.set("");
            }
            KeyCode::Backspace | KeyCode::Delete => {}
            _ => return false,
        }
        true
    }

    /// The part of the text to show in a field `width` characters wide, and
    /// where the cursor is inside it. The cursor is always visible.
    pub fn window(&self, width: usize) -> (String, usize) {
        let width = width.max(2);
        let start = (self.cursor + 1).saturating_sub(width);
        let shown: String = self.text.chars().skip(start).take(width).collect();
        (shown, self.cursor - start)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(code: KeyCode) -> KeyEvent {
        KeyEvent::new(code, KeyModifiers::NONE)
    }

    fn type_text(input: &mut TextInput, text: &str) {
        for c in text.chars() {
            input.handle_key(&key(KeyCode::Char(c)));
        }
    }

    #[test]
    fn typing_and_editing_in_the_middle() {
        let mut input = TextInput::default();
        type_text(&mut input, "helo");
        input.handle_key(&key(KeyCode::Left));
        input.handle_key(&key(KeyCode::Char('l')));
        assert_eq!(input.text(), "hello");
        input.handle_key(&key(KeyCode::Backspace));
        input.handle_key(&key(KeyCode::Delete));
        assert_eq!(input.text(), "hel");
        input.handle_key(&key(KeyCode::Home));
        input.handle_key(&key(KeyCode::Delete));
        assert_eq!((input.text(), input.cursor()), ("el", 0));
    }

    #[test]
    fn works_with_characters_wider_than_a_byte() {
        let mut input = TextInput::default();
        type_text(&mut input, "سلام");
        input.handle_key(&key(KeyCode::Backspace));
        assert_eq!(input.text(), "سلا");
        assert_eq!(input.cursor(), 3);
    }

    #[test]
    fn pasting_drops_line_breaks() {
        let mut input = TextInput::default();
        input.insert("/tmp/a.mp3\n");
        assert_eq!(input.text(), "/tmp/a.mp3");
    }

    #[test]
    fn control_u_clears_and_other_control_keys_are_left_alone() {
        let mut input = TextInput::new("abc");
        assert!(!input.handle_key(&KeyEvent::new(KeyCode::Char('s'), KeyModifiers::CONTROL)));
        assert_eq!(input.text(), "abc");
        assert!(input.handle_key(&KeyEvent::new(KeyCode::Char('u'), KeyModifiers::CONTROL)));
        assert_eq!(input.text(), "");
        assert!(!input.handle_key(&key(KeyCode::Tab)));
    }

    #[test]
    fn long_text_scrolls_to_keep_the_cursor_visible() {
        let input = TextInput::new("0123456789");
        assert_eq!(input.window(20), ("0123456789".to_string(), 10));
        assert_eq!(input.window(5), ("6789".to_string(), 4));
        let mut early = input.clone();
        early.handle_key(&key(KeyCode::Home));
        assert_eq!(early.window(5), ("01234".to_string(), 0));
    }
}
