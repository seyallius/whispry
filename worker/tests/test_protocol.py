import io

import pytest

from helpers import audio, encode
from whispry_worker.protocol import read_message, write_message


def test_round_trip_with_payload():
    payload = audio(5)
    stream = io.BytesIO(encode({"type": "transcribe", "id": 1}, payload) + encode({"type": "cancel", "id": 1}))
    message, data = read_message(stream)
    assert message["type"] == "transcribe" and data == payload
    message, data = read_message(stream)
    assert message["type"] == "cancel" and data is None
    assert read_message(stream) is None


def test_payload_read_in_pieces():
    class Trickle(io.BytesIO):
        def read(self, n=-1):
            return super().read(min(n, 3))

    payload = bytes(range(20))
    message, data = read_message(Trickle(encode({"type": "t"}, payload)))
    assert data == payload


def test_short_read_raises():
    with pytest.raises(EOFError):
        read_message(io.BytesIO(encode({"type": "t"}, b"12345678")[:-3]))


def test_malformed_raises():
    with pytest.raises(ValueError):
        read_message(io.BytesIO(b"{oops\n"))
    with pytest.raises(ValueError):
        read_message(io.BytesIO(b"[1]\n"))


def test_write_message_is_compact_utf8():
    out = io.BytesIO()
    write_message(out, {"type": "segment", "text": "héllo"})
    assert out.getvalue() == '{"type":"segment","text":"héllo"}\n'.encode("utf-8")
