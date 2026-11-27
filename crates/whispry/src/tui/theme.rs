//! Look and feel shared by every screen: palette, gradients, buttons, the
//! logo, and the click targets that make the mouse work.
//!
//! Click targets are recorded while drawing (`add`) and looked up when the
//! mouse is used (`at`), so what you can click is always exactly what is
//! on screen. The last thing drawn wins, so popups sit on top.

use std::cell::{Cell, RefCell};
use std::sync::OnceLock;

use ratatui::Frame;
use ratatui::layout::{Position, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line, Span};
use ratatui::widgets::{BorderType, Paragraph};

use super::run::Button;
use super::setup::Field;

// ----- palette -----

/// Terminals without true colour get the nearest of the 256 fixed colours.
fn truecolor() -> bool {
    static TRUE: OnceLock<bool> = OnceLock::new();
    *TRUE.get_or_init(|| {
        let has = |name: &str, needle: &[&str]| {
            std::env::var(name).is_ok_and(|v| needle.iter().any(|n| v.contains(n)))
        };
        has("COLORTERM", &["truecolor", "24bit"]) || std::env::var_os("WT_SESSION").is_some()
    })
}

pub fn rgb(r: u8, g: u8, b: u8) -> Color {
    if truecolor() {
        return Color::Rgb(r, g, b);
    }
    let level = |v: u8| -> u8 {
        match v {
            0..48 => 0,
            48..115 => 1,
            _ => (v - 35) / 40,
        }
    };
    Color::Indexed(16 + 36 * level(r) + 6 * level(g) + level(b))
}

type Rgb = (u8, u8, u8);

pub struct Palette {
    pub name: &'static str,
    accent: Rgb,
    accent2: Rgb,
    ok: Rgb,
    warn: Rgb,
    error: Rgb,
    dim: Rgb,
    surface: Rgb,
    ink: Rgb,
    /// The gradient runs from `from` to `to`.
    from: Rgb,
    to: Rgb,
}

const SEMANTIC: (Rgb, Rgb, Rgb) = ((110, 231, 183), (251, 191, 36), (251, 113, 133));

const fn palette(
    name: &'static str,
    accent: Rgb,
    accent2: Rgb,
    surface: Rgb,
    from: Rgb,
    to: Rgb,
) -> Palette {
    Palette {
        name,
        accent,
        accent2,
        ok: SEMANTIC.0,
        warn: SEMANTIC.1,
        error: SEMANTIC.2,
        dim: (100, 116, 139),
        surface,
        ink: (17, 17, 27),
        from,
        to,
    }
}

pub const THEMES: [Palette; 6] = [
    palette(
        "Aurora",
        (167, 139, 250),
        (45, 212, 191),
        (46, 43, 74),
        (167, 139, 250),
        (45, 212, 191),
    ),
    palette(
        "Sunset",
        (251, 146, 60),
        (244, 114, 182),
        (68, 40, 40),
        (251, 191, 36),
        (244, 114, 182),
    ),
    palette(
        "Ocean",
        (96, 165, 250),
        (34, 211, 238),
        (30, 48, 80),
        (96, 165, 250),
        (45, 212, 191),
    ),
    palette(
        "Forest",
        (74, 222, 128),
        (190, 242, 100),
        (30, 58, 40),
        (52, 211, 153),
        (190, 242, 100),
    ),
    palette(
        "Neon",
        (255, 64, 200),
        (0, 229, 255),
        (56, 24, 72),
        (255, 64, 200),
        (0, 229, 255),
    ),
    palette(
        "Mono",
        (226, 232, 240),
        (148, 163, 184),
        (51, 56, 66),
        (248, 250, 252),
        (120, 130, 145),
    ),
];

thread_local! {
    static CURRENT: Cell<usize> = const { Cell::new(0) };
}

fn pal() -> &'static Palette {
    &THEMES[CURRENT.get()]
}

pub fn theme_name() -> &'static str {
    pal().name
}

/// Switch to the next theme and remember the choice.
pub fn next_theme() {
    set_theme((CURRENT.get() + 1) % THEMES.len());
    save_theme();
}

fn set_theme(index: usize) {
    CURRENT.set(index % THEMES.len());
}

fn theme_file() -> Option<std::path::PathBuf> {
    let base = std::env::var_os("XDG_CONFIG_HOME")
        .or_else(|| std::env::var_os("APPDATA"))
        .map(std::path::PathBuf::from)
        .or_else(|| std::env::home_dir().map(|home| home.join(".config")))?;
    Some(base.join("whispry").join("theme"))
}

fn save_theme() {
    if cfg!(test) {
        return; // tests must not touch the real config
    }
    if let Some(file) = theme_file() {
        let _ = file.parent().map(std::fs::create_dir_all);
        let _ = std::fs::write(file, pal().name);
    }
}

/// Use the theme chosen last time, if any.
pub fn load_theme() {
    let saved = theme_file().and_then(|file| std::fs::read_to_string(file).ok());
    if let Some(index) = saved.and_then(|name| {
        THEMES
            .iter()
            .position(|t| t.name.eq_ignore_ascii_case(name.trim()))
    }) {
        set_theme(index);
    }
}

fn color(c: Rgb) -> Color {
    rgb(c.0, c.1, c.2)
}

pub fn accent() -> Color {
    color(pal().accent)
}
pub fn accent2() -> Color {
    color(pal().accent2)
}
pub fn dim() -> Color {
    color(pal().dim)
}
pub fn ok() -> Color {
    color(pal().ok)
}
pub fn warn() -> Color {
    color(pal().warn)
}
pub fn error() -> Color {
    color(pal().error)
}
pub fn surface() -> Color {
    color(pal().surface)
}
pub fn ink() -> Color {
    color(pal().ink)
}

pub const BORDER: BorderType = BorderType::Rounded;

pub fn dim_style() -> Style {
    Style::new().fg(dim())
}

/// The theme's first colour at 0, its last at 1.
pub fn gradient(t: f64) -> Color {
    let t = t.clamp(0.0, 1.0);
    let (from, to) = (pal().from, pal().to);
    let mix = |a: u8, b: u8| (f64::from(a) + (f64::from(b) - f64::from(a)) * t).round() as u8;
    rgb(mix(from.0, to.0), mix(from.1, to.1), mix(from.2, to.2))
}

/// Border and title style of a box that may have focus.
pub fn focus_style(focused: bool) -> Style {
    if focused {
        Style::new().fg(accent())
    } else {
        dim_style()
    }
}

// ----- logo -----

const LOGO: [[&str; 3]; 7] = [
    ["█   █", "█ █ █", "▀█▀█▀"],
    ["█   █", "█████", "█   █"],
    ["▀█▀", " █ ", "▄█▄"],
    ["▄▀▀▀▀", "▀▀▀▀▄", "▄▄▄▄▀"],
    ["█▀▀▀▄", "█▀▀▀ ", "█    "],
    ["█▀▀▀▄", "█▀▀█▀", "█   █"],
    ["█   █", " ▀█▀ ", "  █  "],
];

pub const LOGO_WIDTH: u16 = 39;

/// "WHISPRY" in block letters, three rows tall, fading violet to teal.
pub fn logo() -> Vec<Line<'static>> {
    (0..3)
        .map(|row| {
            let mut x = 0;
            let mut spans = Vec::new();
            for letter in LOGO {
                for ch in letter[row].chars().chain([' ']) {
                    let t = f64::from(x) / f64::from(LOGO_WIDTH - 1);
                    spans.push(Span::styled(ch.to_string(), Style::new().fg(gradient(t))));
                    x += 1;
                }
            }
            Line::from(spans)
        })
        .collect()
}

/// A word with the gradient spread over its letters.
pub fn gradient_text(text: &str) -> Vec<Span<'static>> {
    let n = text.chars().count().max(2) - 1;
    text.chars()
        .enumerate()
        .map(|(i, ch)| {
            Span::styled(
                ch.to_string(),
                Style::new()
                    .fg(gradient(i as f64 / n as f64))
                    .add_modifier(Modifier::BOLD),
            )
        })
        .collect()
}

// ----- motion (drawn from elapsed milliseconds) -----

pub fn spinner(ms: u128) -> &'static str {
    const FRAMES: [&str; 10] = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"];
    FRAMES[(ms / 80) as usize % FRAMES.len()]
}

/// A little bouncing equalizer: speech, visualised.
pub fn equalizer(ms: u128, bars: usize) -> Vec<Span<'static>> {
    const LEVELS: [&str; 8] = ["▁", "▂", "▃", "▄", "▅", "▆", "▇", "█"];
    (0..bars)
        .map(|i| {
            let phase = ms as f64 / 170.0 + i as f64 * 0.85 + (i * i) as f64 * 0.13;
            let level = ((phase.sin() + 1.0) / 2.0 * 7.0).round() as usize;
            Span::styled(
                LEVELS[level.min(7)],
                Style::new().fg(gradient(i as f64 / bars.max(2) as f64)),
            )
        })
        .collect()
}

/// A progress bar of `width` cells. Without a ratio a bright patch sweeps
/// back and forth instead.
pub fn progress_bar(width: usize, ratio: Option<f64>, done: bool, ms: u128) -> Vec<Span<'static>> {
    let track = Style::new().fg(rgb(55, 65, 81));
    (0..width)
        .map(|i| {
            let t = i as f64 / width.max(2) as f64;
            let lit = match ratio {
                _ if done => true,
                Some(r) => (i as f64) < r * width as f64,
                None => {
                    let span = (width + 8) as u128;
                    let head = (ms / 40 % (span * 2)) as usize;
                    let head = if head < span as usize {
                        head
                    } else {
                        span as usize * 2 - head
                    };
                    i + 8 > head && i <= head
                }
            };
            if lit {
                let color = if done { ok() } else { gradient(t) };
                Span::styled("━", Style::new().fg(color))
            } else {
                Span::styled("─", track)
            }
        })
        .collect()
}

// ----- click targets -----

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Hit {
    /// A setup field or button: give it focus (and press it, if it is one).
    Setup(Field),
    /// The ◀ (false) or ▶ (true) of a setup choice.
    Arrow(Field, bool),
    Device(usize),
    Browse,
    Run(Button),
    PickerRow(usize),
    /// A folder in the path shown above the picker's list (0 is the root).
    PickerCrumb(usize),
    PickerOutside,
    Help,
    HelpClose,
    Theme,
    Dialog(usize),
    /// Part of a popup that does nothing but keep clicks from passing through.
    Nothing,
}

thread_local! {
    static HITS: RefCell<Vec<(Rect, Hit)>> = const { RefCell::new(Vec::new()) };
    static MOUSE: Cell<Option<Position>> = const { Cell::new(None) };
}

pub fn clear_hits() {
    HITS.with_borrow_mut(Vec::clear);
}

pub fn add(rect: Rect, hit: Hit) {
    HITS.with_borrow_mut(|hits| hits.push((rect, hit)));
}

pub fn at(x: u16, y: u16) -> Option<Hit> {
    HITS.with_borrow(|hits| {
        hits.iter()
            .rev()
            .find(|(rect, _)| rect.contains(Position::new(x, y)))
            .map(|(_, hit)| *hit)
    })
}

pub fn set_mouse(x: u16, y: u16) {
    MOUSE.set(Some(Position::new(x, y)));
}

pub fn hovered(rect: Rect) -> bool {
    MOUSE.get().is_some_and(|pos| rect.contains(pos))
}

#[cfg(test)]
pub fn rect_of(hit: Hit) -> Option<Rect> {
    HITS.with_borrow(|hits| hits.iter().rev().find(|(_, h)| *h == hit).map(|(r, _)| *r))
}

// ----- buttons -----

/// Draw a row of buttons from (`x`, `y`) and make each clickable. Returns the
/// width used.
pub fn buttons(frame: &mut Frame, x: u16, y: u16, items: &[(&str, Hit, bool)]) -> u16 {
    let mut at_x = x;
    let right = frame.area().right();
    for (label, hit, focused) in items {
        let width = label.chars().count() as u16 + 4;
        if at_x + width > right {
            break;
        }
        let rect = Rect::new(at_x, y, width, 1);
        let bg = if *focused {
            accent()
        } else if hovered(rect) {
            accent2()
        } else {
            surface()
        };
        let lit = *focused || hovered(rect);
        let body = if lit {
            Style::new().fg(ink()).bg(bg).add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(accent()).bg(bg)
        };
        let cap = Style::new().fg(bg);
        frame.render_widget(
            Paragraph::new(Line::from(vec![
                Span::styled("▐", cap),
                Span::styled(format!(" {label} "), body),
                Span::styled("▌", cap),
            ])),
            rect,
        );
        add(rect, *hit);
        at_x += width + 2;
    }
    at_x.saturating_sub(x + 2)
}

/// "? Help · ◐ Aurora", clickable, right-aligned on the row `area`.
pub fn chips(frame: &mut Frame, area: Rect) {
    let help = " ? Help ";
    let theme = format!(" ◐ {} ", theme_name());
    let width = (help.chars().count() + theme.chars().count() + 1) as u16;
    if area.width < width + 20 {
        return;
    }
    let x = area.right() - width;
    for (text, hit, at_x) in [
        (help.to_string(), Hit::Help, x),
        (
            theme.clone(),
            Hit::Theme,
            x + help.chars().count() as u16 + 1,
        ),
    ] {
        let rect = Rect::new(at_x, area.y, text.chars().count() as u16, 1);
        let style = if hovered(rect) {
            Style::new()
                .fg(ink())
                .bg(accent2())
                .add_modifier(Modifier::BOLD)
        } else {
            Style::new().fg(accent()).bg(surface())
        };
        frame.render_widget(Paragraph::new(Span::styled(text, style)), rect);
        add(rect, hit);
    }
}
