//! Decode audio with FFmpeg, one block at a time.

use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Stdio};

use crate::config::SAMPLE_RATE;
use crate::ffmpeg::FFmpeg;
use crate::process::command;

pub const BYTES_PER_SAMPLE: usize = 4;

/// How much of FFmpeg's error output is kept for the message.
const ERROR_DETAIL_BYTES: u64 = 1000;

#[derive(Debug, thiserror::Error)]
pub enum AudioDecodeError {
    #[error("Could not start FFmpeg: {0}")]
    Start(std::io::Error),
    #[error("Could not read decoded audio: {0}")]
    Read(std::io::Error),
    #[error("FFmpeg could not decode {file} (exit code {code}): {detail}")]
    Failed {
        file: String,
        code: String,
        detail: String,
    },
}

/// 16 kHz mono float32 blocks decoded from a media file.
///
/// Only one block is held in memory at a time. Dropping the stream early
/// stops FFmpeg.
pub struct AudioStream {
    child: Child,
    stdout: ChildStdout,
    // FFmpeg's stderr goes to a temp file so a chatty FFmpeg can never block
    // on a full pipe.
    errors: File,
    file: PathBuf,
    block_bytes: usize,
    done: bool,
}

/// Start decoding `audio`, yielding blocks of `block_seconds` (the last may
/// be shorter).
pub fn stream_audio(
    audio: &Path,
    tool: &FFmpeg,
    block_seconds: f64,
) -> Result<AudioStream, AudioDecodeError> {
    let errors = tempfile::tempfile().map_err(AudioDecodeError::Start)?;
    let mut child = command(&tool.ffmpeg)
        .args(["-nostdin", "-v", "error", "-i"])
        .arg(audio)
        .args(["-f", "f32le", "-ac", "1", "-ar"])
        .arg(SAMPLE_RATE.to_string())
        .arg("pipe:1")
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::from(
            errors.try_clone().map_err(AudioDecodeError::Start)?,
        ))
        .spawn()
        .map_err(AudioDecodeError::Start)?;
    let stdout = child.stdout.take().expect("stdout was piped");

    let samples = ((block_seconds * f64::from(SAMPLE_RATE)) as usize).max(1);
    Ok(AudioStream {
        child,
        stdout,
        errors,
        file: audio.to_path_buf(),
        block_bytes: samples * BYTES_PER_SAMPLE,
        done: false,
    })
}

impl AudioStream {
    /// Fill the buffer from the pipe; shorter only at the end of the audio.
    fn read_block(&mut self) -> std::io::Result<Vec<u8>> {
        let mut buffer = vec![0u8; self.block_bytes];
        let mut filled = 0;
        while filled < buffer.len() {
            match self.stdout.read(&mut buffer[filled..]) {
                Ok(0) => break,
                Ok(count) => filled += count,
                Err(error) if error.kind() == std::io::ErrorKind::Interrupted => {}
                Err(error) => return Err(error),
            }
        }
        buffer.truncate(filled);
        Ok(buffer)
    }

    /// Called at the end of the audio: report a failed FFmpeg run.
    fn finish(&mut self) -> Result<(), AudioDecodeError> {
        let status = self.child.wait().map_err(AudioDecodeError::Read)?;
        if status.success() {
            return Ok(());
        }
        Err(AudioDecodeError::Failed {
            file: self.file.file_name().map_or_else(
                || self.file.display().to_string(),
                |n| n.to_string_lossy().into(),
            ),
            code: status
                .code()
                .map_or_else(|| "unknown".into(), |c| c.to_string()),
            detail: self.error_detail(),
        })
    }

    fn error_detail(&mut self) -> String {
        let mut text = Vec::new();
        let read = self
            .errors
            .seek(SeekFrom::Start(0))
            .and_then(|_| self.errors.try_clone())
            .and_then(|file| file.take(ERROR_DETAIL_BYTES * 8).read_to_end(&mut text));
        if read.is_err() {
            return String::new();
        }
        let text = String::from_utf8_lossy(&text);
        let text = text.trim();
        let keep_from = text
            .char_indices()
            .rev()
            .nth(ERROR_DETAIL_BYTES as usize - 1)
            .map_or(0, |(index, _)| index);
        text[keep_from..].to_string()
    }
}

impl Iterator for AudioStream {
    type Item = Result<Vec<f32>, AudioDecodeError>;

    fn next(&mut self) -> Option<Self::Item> {
        if self.done {
            return None;
        }
        let bytes = match self.read_block() {
            Ok(bytes) => bytes,
            Err(error) => {
                self.done = true;
                return Some(Err(AudioDecodeError::Read(error)));
            }
        };

        if bytes.len() < self.block_bytes {
            // The pipe is closed: this is the last block (possibly empty).
            self.done = true;
            if let Err(error) = self.finish() {
                return Some(Err(error));
            }
        }

        // A trailing partial sample, if any, is dropped.
        let samples: Vec<f32> = bytes
            .chunks_exact(BYTES_PER_SAMPLE)
            .map(|chunk| f32::from_le_bytes(chunk.try_into().expect("chunk is 4 bytes")))
            .collect();
        if samples.is_empty() && self.done {
            return None;
        }
        Some(Ok(samples))
    }
}

impl Drop for AudioStream {
    fn drop(&mut self) {
        if !self.done {
            let _ = self.child.kill();
            let _ = self.child.wait();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ffmpeg::find_ffmpeg;
    use crate::test_support::write_sine_wav;

    fn rate() -> usize {
        SAMPLE_RATE as usize
    }

    #[test]
    fn yields_bounded_blocks_covering_the_audio() {
        let Some(tool) = find_ffmpeg() else { return };
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("sine.wav");
        write_sine_wav(&wav, 3, 440.0);

        let blocks: Vec<Vec<f32>> = stream_audio(&wav, &tool, 1.0)
            .unwrap()
            .collect::<Result<_, _>>()
            .unwrap();
        assert!(blocks.iter().all(|block| block.len() <= rate()));
        let total: usize = blocks.iter().map(Vec::len).sum();
        assert!(total.abs_diff(3 * rate()) < rate() / 100, "{total}");
        assert!(blocks[0].iter().any(|sample| sample.abs() > 0.1));
    }

    #[test]
    fn reports_decode_errors_with_the_file_name() {
        let Some(tool) = find_ffmpeg() else { return };
        let dir = tempfile::tempdir().unwrap();
        let broken = dir.path().join("broken.mp3");
        std::fs::write(&broken, b"not audio").unwrap();

        let result: Result<Vec<_>, _> = stream_audio(&broken, &tool, 30.0).unwrap().collect();
        let message = result.unwrap_err().to_string();
        assert!(message.contains("broken.mp3"), "{message}");
    }

    #[test]
    fn dropping_the_stream_early_stops_ffmpeg() {
        let Some(tool) = find_ffmpeg() else { return };
        let dir = tempfile::tempdir().unwrap();
        let wav = dir.path().join("long.wav");
        write_sine_wav(&wav, 120, 440.0);

        let mut stream = stream_audio(&wav, &tool, 1.0).unwrap();
        assert!(stream.next().unwrap().is_ok());
        drop(stream); // must not hang or leave FFmpeg running
    }

    #[test]
    fn a_missing_ffmpeg_binary_is_a_start_error() {
        let tool = FFmpeg {
            ffmpeg: PathBuf::from("/nonexistent/ffmpeg"),
            ffprobe: None,
        };
        let error = stream_audio(Path::new("a.wav"), &tool, 1.0).err().unwrap();
        assert!(error.to_string().contains("Could not start FFmpeg"));
    }
}
