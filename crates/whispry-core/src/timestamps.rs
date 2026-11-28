//! Formatting of positions in the audio.

/// Format seconds as `MM:SS`, or `HH:MM:SS` from one hour on. Fractions are
/// dropped and negative values count as zero.
pub fn format_timestamp(seconds: f64) -> String {
    let total = seconds.max(0.0) as u64;
    let (hours, minutes, secs) = (total / 3600, total % 3600 / 60, total % 60);
    if hours > 0 {
        format!("{hours:02}:{minutes:02}:{secs:02}")
    } else {
        format!("{minutes:02}:{secs:02}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn formats_short_and_long_positions() {
        assert_eq!(format_timestamp(0.0), "00:00");
        assert_eq!(format_timestamp(65.9), "01:05");
        assert_eq!(format_timestamp(3725.0), "01:02:05");
    }

    #[test]
    fn negative_counts_as_zero() {
        assert_eq!(format_timestamp(-4.0), "00:00");
    }
}
