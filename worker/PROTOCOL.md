# Worker protocol

The Rust app starts one worker process per job and talks to it over the
worker's standard input and output. The worker keeps the model loaded
between audio windows.

    python -m whispry_worker

## Framing

Every message is one JSON object on one line, UTF-8, ending in `\n`.
A message may be followed by raw bytes; if it is, its `audio_bytes` field
says how many. Standard output carries only these messages. Anything else
the worker (or a library it uses) prints goes to standard error.

Times are seconds, relative to the start of the window being transcribed.
The Rust side adds the window's offset.

## Rust → worker

| Message | Fields | Meaning |
|---|---|---|
| `load` | `model_dir`, `device` (`"cpu"` or `"cuda"`), `compute_types`, `cpu_threads` | Load the model. `compute_types` lists the compute types to try, best first; the worker uses the first one the device supports. Replies `loaded` or `error`. |
| `transcribe` | `id`, `language` (string or `null`), `beam_size`, `initial_prompt` (string or `null`), `audio_bytes` | Followed by `audio_bytes` bytes of 16 kHz mono little-endian float32 samples. Replies with `language`, any number of `segment`, then `done`. |
| `cancel` | `id` | Stop the running `transcribe` with this id after the current segment. It still replies `done` with `cancelled: true`. Ignored if that id is not running. |
| `shutdown` | | Exit. Closing standard input does the same. |

## Worker → Rust

| Message | Fields | Meaning |
|---|---|---|
| `ready` | `version`, `faster_whisper` | Sent once when the worker has started. |
| `loaded` | `device`, `compute_type` | The model is ready. `compute_type` is the one that was chosen. |
| `language` | `id`, `language`, `probability` | The language faster-whisper detected or was given for this window. |
| `segment` | `id`, `start`, `end`, `text` | One transcribed segment, sent as soon as it is ready. |
| `done` | `id`, `cancelled` | The window is finished. |
| `error` | `id` (number or `null`), `stage` (`"load"`, `"transcribe"` or `"protocol"`), `message` | Something failed. After a `load` error the worker stays alive but has no model. After a `transcribe` error it stays alive and loaded. After a `protocol` error it exits. |

## Ordering

* The worker handles `load` and `transcribe` one at a time, in the order
  they arrive. `cancel` and `shutdown` are read while a window is running.
* Speech detection (`vad_filter`) is always on.
* Segment text is stripped of leading and trailing whitespace.
