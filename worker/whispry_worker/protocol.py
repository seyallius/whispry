"""Line-delimited JSON messages with optional raw byte payloads."""

import json
import threading

_write_lock = threading.Lock()


def read_message(stream):
    """Read one message; returns (message, audio_bytes_or_None), or None at EOF."""
    line = stream.readline()
    if not line:
        return None
    message = json.loads(line)
    if not isinstance(message, dict):
        raise ValueError("message is not a JSON object")
    size = message.get("audio_bytes")
    if size is None:
        return message, None
    if not isinstance(size, int) or isinstance(size, bool) or size < 0:
        raise ValueError("invalid audio_bytes")
    chunks = []
    remaining = size
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"expected {size} audio bytes, got {size - remaining}")
        chunks.append(chunk)
        remaining -= len(chunk)
    return message, b"".join(chunks)


def write_message(stream, message):
    """Write one message as a compact JSON line and flush."""
    data = json.dumps(message, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    with _write_lock:
        stream.write(data + b"\n")
        stream.flush()
