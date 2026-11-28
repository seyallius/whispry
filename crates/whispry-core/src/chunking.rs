//! Split a stream of audio blocks into windows that end in silence.
//!
//! Each window is transcribed on its own, so memory use is bounded by the
//! window length instead of the file length. Cutting inside a pause keeps
//! words from being split across two windows.

use crate::config::SAMPLE_RATE;

/// 20 ms of audio.
pub const FRAME: usize = SAMPLE_RATE as usize / 50;

/// A quiet stretch must last this many frames (100 ms) to count as a pause.
const MIN_PAUSE_FRAMES: usize = 5;

/// How far above the noise floor, as a share of the gap between the noise
/// floor and the speech level, a frame may be and still count as quiet.
const QUIET_SHARE: f32 = 0.15;

/// Below this level (RMS) a region is treated as silence.
const SILENCE_LEVEL: f32 = 1e-4;

#[derive(Clone, Debug, PartialEq)]
pub struct AudioWindow {
    /// Index of the first sample in the whole recording.
    pub offset: usize,
    pub samples: Vec<f32>,
}

impl AudioWindow {
    /// Where the window starts, in seconds.
    pub fn start(&self) -> f64 {
        self.offset as f64 / f64::from(SAMPLE_RATE)
    }

    pub fn duration(&self) -> f64 {
        self.samples.len() as f64 / f64::from(SAMPLE_RATE)
    }
}

/// Root-mean-square level of each whole 20 ms frame.
fn frame_levels(samples: &[f32]) -> Vec<f32> {
    samples
        .chunks_exact(FRAME)
        .map(|frame| (frame.iter().map(|s| s * s).sum::<f32>() / FRAME as f32).sqrt())
        .collect()
}

/// Middle of the quietest 20 ms frame. Fallback when there is no pause.
pub fn quietest_point(samples: &[f32]) -> usize {
    let levels = frame_levels(samples);
    levels
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.total_cmp(b.1))
        .map_or(samples.len(), |(index, _)| index * FRAME + FRAME / 2)
}

/// The first and one-past-last frame of the longest run of quiet frames.
fn longest_quiet_run(quiet: &[bool]) -> Option<(usize, usize)> {
    let mut best: Option<(usize, usize)> = None;
    let mut start = 0;
    while start < quiet.len() {
        if !quiet[start] {
            start += 1;
            continue;
        }
        let end = quiet[start..]
            .iter()
            .position(|q| !q)
            .map_or(quiet.len(), |n| start + n);
        // Ties go to the later run, which makes a longer window.
        if best.is_none_or(|(s, e)| end - start >= e - s) {
            best = Some((start, end));
        }
        start = end;
    }
    best
}

/// Pick where to end a window inside `samples` (the search region).
///
/// Cuts in the middle of the longest pause. A silent region is cut at its
/// end; with no pause at all, the cut goes to the quietest moment.
pub fn find_cut(samples: &[f32]) -> usize {
    let levels = frame_levels(samples);
    if levels.is_empty() {
        return samples.len();
    }

    let mut sorted = levels.clone();
    sorted.sort_by(f32::total_cmp);
    let at = |share: f32| sorted[((sorted.len() - 1) as f32 * share) as usize];
    let (floor, speech) = (at(0.1), at(0.9));

    if speech < SILENCE_LEVEL {
        return samples.len();
    }
    if speech < floor * 2.0 {
        // Steady sound with no pauses.
        return quietest_point(samples);
    }

    let threshold = floor + QUIET_SHARE * (speech - floor);
    let quiet: Vec<bool> = levels.iter().map(|level| *level <= threshold).collect();
    match longest_quiet_run(&quiet) {
        Some((start, end)) if end - start >= MIN_PAUSE_FRAMES => (start + end) * FRAME / 2,
        _ => quietest_point(samples),
    }
}

/// Groups decoded blocks into windows. Created by [`iter_windows`].
pub struct Windows<I, F> {
    blocks: I,
    find_cut: F,
    window: usize,
    search: usize,
    buffer: Vec<f32>,
    offset: usize,
    finished: bool,
}

/// Group decoded blocks into windows of about `window_seconds`.
///
/// Each window ends somewhere in its last `search_seconds`, at a pause found
/// by `find_cut`. At most one window plus one block is held in memory.
pub fn iter_windows<I, E>(
    blocks: I,
    window_seconds: f64,
    search_seconds: f64,
    find_cut: impl FnMut(&[f32]) -> usize,
) -> Windows<I::IntoIter, impl FnMut(&[f32]) -> usize>
where
    I: IntoIterator<Item = Result<Vec<f32>, E>>,
{
    let window = (window_seconds * f64::from(SAMPLE_RATE)) as usize;
    let search = ((search_seconds * f64::from(SAMPLE_RATE)) as usize).min(window / 2);
    Windows {
        blocks: blocks.into_iter(),
        find_cut,
        window,
        search,
        buffer: Vec::new(),
        offset: 0,
        finished: false,
    }
}

impl<I, E, F> Iterator for Windows<I, F>
where
    I: Iterator<Item = Result<Vec<f32>, E>>,
    F: FnMut(&[f32]) -> usize,
{
    type Item = Result<AudioWindow, E>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.finished {
            return None;
        }
        while self.buffer.len() < self.window {
            match self.blocks.next() {
                Some(Ok(block)) => self.buffer.extend_from_slice(&block),
                Some(Err(error)) => {
                    self.finished = true;
                    return Some(Err(error));
                }
                None => {
                    self.finished = true;
                    if self.buffer.is_empty() {
                        return None;
                    }
                    let samples = std::mem::take(&mut self.buffer);
                    return Some(Ok(AudioWindow {
                        offset: self.offset,
                        samples,
                    }));
                }
            }
        }

        let region_start = self.window - self.search;
        let region = &self.buffer[region_start..self.window];
        let cut = (region_start + (self.find_cut)(region)).max(1);
        let samples: Vec<f32> = self.buffer.drain(..cut).collect();
        let window = AudioWindow {
            offset: self.offset,
            samples,
        };
        self.offset += cut;
        Some(Ok(window))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::convert::Infallible;

    const RATE: usize = SAMPLE_RATE as usize;

    fn seconds(count: f64) -> usize {
        (count * RATE as f64) as usize
    }

    fn blocks_of(samples: &[f32], size: usize) -> Vec<Result<Vec<f32>, Infallible>> {
        samples
            .chunks(size)
            .map(|chunk| Ok(chunk.to_vec()))
            .collect()
    }

    /// A loud tone with silence between `from` and `to` (in seconds).
    fn tone_with_pauses(length: f64, pauses: &[(f64, f64)]) -> Vec<f32> {
        (0..seconds(length))
            .map(|n| {
                let time = n as f64 / RATE as f64;
                if pauses.iter().any(|(from, to)| (*from..*to).contains(&time)) {
                    0.0
                } else {
                    (time * 440.0 * std::f64::consts::TAU).sin() as f32 * 0.5
                }
            })
            .collect()
    }

    fn collect(windows: impl Iterator<Item = Result<AudioWindow, Infallible>>) -> Vec<AudioWindow> {
        windows.map(Result::unwrap).collect()
    }

    #[test]
    fn windows_cover_the_audio_exactly_once() {
        let audio: Vec<f32> = (0..seconds(25.0)).map(|n| n as f32).collect();
        let windows = collect(iter_windows(blocks_of(&audio, 3 * RATE), 10.0, 2.0, |s| {
            s.len()
        }));

        let durations: Vec<f64> = windows.iter().map(AudioWindow::duration).collect();
        let starts: Vec<f64> = windows.iter().map(AudioWindow::start).collect();
        assert_eq!(durations, [10.0, 10.0, 5.0]);
        assert_eq!(starts, [0.0, 10.0, 20.0]);
        let joined: Vec<f32> = windows.into_iter().flat_map(|w| w.samples).collect();
        assert_eq!(joined, audio);
    }

    #[test]
    fn windows_never_exceed_the_configured_length() {
        let audio = vec![0.0f32; seconds(120.0)];
        for window in collect(iter_windows(blocks_of(&audio, RATE), 10.0, 2.0, |s| {
            s.len()
        })) {
            assert!(window.samples.len() <= seconds(10.0));
        }
    }

    #[test]
    fn an_empty_stream_yields_no_windows() {
        let none: Vec<Result<Vec<f32>, Infallible>> = Vec::new();
        assert!(collect(iter_windows(none, 10.0, 2.0, |s| s.len())).is_empty());
    }

    #[test]
    fn a_decode_error_is_passed_on_and_ends_the_stream() {
        let blocks = vec![Ok(vec![0.0f32; RATE]), Err("broken")];
        let mut windows = iter_windows(blocks, 10.0, 2.0, |s| s.len());
        assert_eq!(windows.next(), Some(Err("broken")));
        assert_eq!(windows.next(), None);
    }

    #[test]
    fn cut_lands_in_the_longest_pause() {
        // 10 s region: tone 0-3 s, pause 3-7 s, tone 7-9 s, short pause 9-10 s.
        let region = tone_with_pauses(10.0, &[(3.0, 7.0), (9.0, 10.0)]);
        let cut = find_cut(&region) as f64 / RATE as f64;
        assert!((cut - 5.0).abs() < 0.05, "{cut}");
    }

    #[test]
    fn cut_falls_back_to_the_quietest_moment_during_nonstop_sound() {
        let mut region = vec![0.5f32; RATE];
        region[8000..8320].fill(0.0); // one quiet 20 ms frame
        let cut = find_cut(&region);
        assert!((8000..=8320).contains(&cut), "{cut}");
        assert_eq!(cut, quietest_point(&region));
    }

    #[test]
    fn a_silent_region_is_cut_at_its_end() {
        let region = vec![0.0f32; 5 * RATE];
        assert_eq!(find_cut(&region), region.len());
    }

    #[test]
    fn a_region_shorter_than_a_frame_is_cut_at_its_end() {
        assert_eq!(find_cut(&[0.5; 10]), 10);
    }

    #[test]
    fn windows_end_inside_pauses() {
        // A pause centred every 9 s, in a 10 s window with a 4 s search.
        let pauses: Vec<(f64, f64)> = (1..=3)
            .map(|k| (9.0 * k as f64 - 0.5, 9.0 * k as f64 + 0.5))
            .collect();
        let audio = tone_with_pauses(30.0, &pauses);
        let windows = collect(iter_windows(blocks_of(&audio, RATE), 10.0, 4.0, find_cut));

        assert!(
            (windows[0].duration() - 9.0).abs() < 0.05,
            "{}",
            windows[0].duration()
        );
        assert!((windows[1].start() - 9.0).abs() < 0.05);
        let total: usize = windows.iter().map(|w| w.samples.len()).sum();
        assert_eq!(total, audio.len());
    }

    #[test]
    fn audio_window_start_and_duration() {
        let window = AudioWindow {
            offset: RATE * 2,
            samples: vec![0.0; RATE / 2],
        };
        assert_eq!((window.start(), window.duration()), (2.0, 0.5));
    }
}
