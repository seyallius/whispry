//! Helpers shared by tests.

use std::path::Path;

/// Write a 16 kHz mono 16-bit WAV file holding a sine wave.
pub fn write_sine_wav(path: &Path, seconds: u32, frequency: f64) {
    const RATE: u32 = 16_000;
    let samples = RATE * seconds;
    let mut bytes = Vec::with_capacity(44 + samples as usize * 2);
    bytes.extend_from_slice(b"RIFF");
    bytes.extend_from_slice(&(36 + samples * 2).to_le_bytes());
    bytes.extend_from_slice(b"WAVEfmt ");
    bytes.extend_from_slice(&16u32.to_le_bytes());
    bytes.extend_from_slice(&1u16.to_le_bytes()); // PCM
    bytes.extend_from_slice(&1u16.to_le_bytes()); // mono
    bytes.extend_from_slice(&RATE.to_le_bytes());
    bytes.extend_from_slice(&(RATE * 2).to_le_bytes());
    bytes.extend_from_slice(&2u16.to_le_bytes());
    bytes.extend_from_slice(&16u16.to_le_bytes());
    bytes.extend_from_slice(b"data");
    bytes.extend_from_slice(&(samples * 2).to_le_bytes());
    for n in 0..samples {
        let phase = 2.0 * std::f64::consts::PI * frequency * f64::from(n) / f64::from(RATE);
        bytes.extend_from_slice(&((phase.sin() * 8000.0) as i16).to_le_bytes());
    }
    std::fs::write(path, bytes).unwrap();
}
