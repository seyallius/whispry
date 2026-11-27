#!/usr/bin/env python3
"""Regenerate docs/screenshots by driving the real Whispry binary.

Runs the TUI in a pseudo-terminal, reads the screen with a terminal emulator
(pyte), and draws it with Pillow. Nothing is mocked: the transcript you see is
a real run of the Small model on a short synthetic recording.

Needs: pip install pyte pillow; ffmpeg and espeak-ng (to make the demo audio);
a cached Small model; and a built binary (cargo build --release).

    python3 scripts/screenshots.py [path/to/whispry] [output_dir]
"""

import fcntl
import os
import pty
import select
import shutil
import struct
import subprocess
import sys
import tempfile
import termios
import time
from pathlib import Path

import pyte
from PIL import Image, ImageDraw, ImageFont

BINARY = sys.argv[1] if len(sys.argv) > 1 else "target/release/whispry"
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "docs/screenshots")
COLS, ROWS = 100, 30
FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"
FONT_BOLD = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Bold.ttf"
FONT_ITALIC = "/usr/share/fonts/truetype/dejavu/DejaVuSansMono-Oblique.ttf"
SIZE = 20
THEMES = ["Aurora", "Sunset", "Ocean", "Forest", "Neon", "Mono", "Paper",
          "Dracula", "Nord", "Gruvbox", "Solarized", "Dawn"]

# DejaVu Sans Mono has no braille (the spinner); plain DejaVu Sans does.
FALLBACK = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", SIZE - 3)
regular = ImageFont.truetype(FONT, SIZE)
bold = ImageFont.truetype(FONT_BOLD, SIZE)
italic = ImageFont.truetype(FONT_ITALIC, SIZE)
ascent, descent = regular.getmetrics()
CW, CH = round(regular.getlength("M")), ascent + descent
BAR = 36  # window title bar


def make_demo(root: Path) -> Path:
    """A fake home with a few recordings, and a ~1 minute spoken recording."""
    home = root / "home"
    for folder in ["Recordings", "Documents", "Music", "Downloads"]:
        (home / folder).mkdir(parents=True, exist_ok=True)
    cache = home / ".cache/whispry"
    cache.mkdir(parents=True, exist_ok=True)
    (cache / "models").symlink_to(Path.home() / ".cache/whispry/models")
    text = ("Welcome everyone, and thanks for joining the weekly planning meeting. "
            "Let's start with the launch date. The design is finished, and the team "
            "is confident we can ship on Thursday. Next, we'll review customer "
            "feedback and decide what to fix first. ")
    subprocess.run(["espeak-ng", "-s", "150", "-w", str(root / "raw.wav"), text * 2], check=True)
    recording = home / "Recordings" / "weekly-planning.wav"
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-i", str(root / "raw.wav"),
                    str(recording)], check=True)
    for name in ["interview-anna.mp3", "lecture-01.m4a", "voice-memo.wav", "standup.mp4"]:
        (home / "Recordings" / name).write_bytes(b"")
    (home / "Recordings" / "2024-notes").mkdir(exist_ok=True)
    return recording


class Session:
    """The TUI in a pseudo-terminal with a screen we can read."""

    def __init__(self, root: Path, theme: str, recording: Path, path=None):
        config = root / "config" / "whispry"
        config.mkdir(parents=True, exist_ok=True)
        (config / "theme").write_text(theme)
        env = dict(os.environ, HOME=str(root / "home"), XDG_CONFIG_HOME=str(root / "config"),
                   COLORTERM="truecolor", TERM="xterm-256color",
                   WHISPRY_MODEL_DIR=str(root / "home/.cache/whispry/models"))
        if path is not None:
            env["PATH"] = path
        self.pid, self.fd = pty.fork()
        if self.pid == 0:
            os.chdir(root / "home" / "Recordings")
            for key, value in env.items():
                os.environ[key] = value
            os.execv(BINARY, [BINARY, "tui", "~/Recordings/weekly-planning.wav"])
        fcntl.ioctl(self.fd, termios.TIOCSWINSZ, struct.pack("HHHH", ROWS, COLS, 0, 0))
        self.screen = pyte.Screen(COLS, ROWS)
        self.stream = pyte.ByteStream(self.screen)
        self.pump(1.2)

    def pump(self, seconds):
        end = time.time() + seconds
        while time.time() < end:
            ready, _, _ = select.select([self.fd], [], [], 0.03)
            if ready:
                try:
                    self.stream.feed(os.read(self.fd, 65536))
                except OSError:
                    return

    def send(self, data: bytes, wait=0.35):
        os.write(self.fd, data)
        self.pump(wait)

    def text(self):
        return "\n".join(self.screen.display)

    def find(self, needle):
        for y, line in enumerate(self.screen.display):
            x = line.find(needle)
            if x >= 0:
                return x, y
        raise LookupError(needle)

    def mouse(self, x, y, kind):
        """kind: 'move', 'down', 'up'. Coordinates are 0-based cells."""
        code = {"move": 35, "down": 0, "up": 0}[kind]
        tail = "m" if kind == "up" else "M"
        self.send(f"\x1b[<{code};{x + 1};{y + 1}{tail}".encode(), 0.15)

    def click_on(self, needle, dx=2):
        x, y = self.find(needle)
        self.mouse(x + dx, y, "move")
        self.mouse(x + dx, y, "down")
        self.mouse(x + dx, y, "up")

    def hover_on(self, needle, dx=2):
        x, y = self.find(needle)
        self.mouse(x + dx, y, "move")

    def close(self):
        try:
            os.write(self.fd, b"\x11\x11")
            time.sleep(0.2)
            os.kill(self.pid, 9)
        except OSError:
            pass
        try:
            os.waitpid(self.pid, 0)
        except OSError:
            pass


def color(value, default):
    if value in ("default", None):
        return default
    if len(value) == 6:
        try:
            return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))
        except ValueError:
            pass
    names = {"black": (0, 0, 0), "white": (229, 229, 229)}
    return names.get(value, default)


def render(session: Session, title="whispry") -> Image.Image:
    """Draw the emulated screen as a window."""
    page = color(session.screen.buffer[0][0].bg, (20, 20, 30))
    width, height = COLS * CW + 2 * 14, ROWS * CH + 2 * 10
    image = Image.new("RGB", (width, height + BAR), page)
    draw = ImageDraw.Draw(image)
    # title bar
    shade = tuple(max(0, c - 14) if sum(page) > 300 else min(255, c + 14) for c in page)
    draw.rectangle([0, 0, width, BAR], fill=shade)
    for i, dot in enumerate([(255, 95, 86), (255, 189, 46), (39, 201, 63)]):
        draw.ellipse([16 + i * 26, 11, 30 + i * 26, 25], fill=dot)
    label = f"{title}"
    draw.text(((width - regular.getlength(label)) / 2, 7), label, fill=(150, 150, 165), font=regular)
    for y in range(ROWS):
        row = session.screen.buffer[y]
        for x in range(COLS):
            char = row[x]
            fg = color(char.fg, (220, 220, 230))
            bg = color(char.bg, page)
            if char.reverse:
                fg, bg = bg, fg
            px, py = 14 + x * CW, BAR + 10 + y * CH
            if bg != page:
                draw.rectangle([px, py, px + CW - 1, py + CH - 1], fill=bg)
            if char.data.strip():
                font = bold if char.bold else italic if char.italics else regular
                if "\u2800" <= char.data <= "\u28ff":
                    font = FALLBACK
                draw.text((px, py), char.data, fill=fg, font=font)
                if char.underscore:
                    draw.line([px, py + CH - 2, px + CW, py + CH - 2], fill=fg)
    return image


def save(image, name):
    OUT.mkdir(parents=True, exist_ok=True)
    image.save(OUT / name, optimize=True)
    print("wrote", OUT / name)


def wait_for(session, needle, timeout=120):
    end = time.time() + timeout
    while time.time() < end:
        session.pump(0.4)
        if needle in session.text():
            return
    raise TimeoutError(needle)


def stills(root, recording):
    s = Session(root, "Aurora", recording)
    save(render(s), "setup.png")
    s.send(b"\x0f", 0.5)  # Ctrl+O
    save(render(s), "browser.png")
    s.send(b"\x1b", 0.3)
    s.send(b"\x1bOP", 0.5)  # F1
    save(render(s), "help.png")
    s.close()

    s = Session(root, "Ocean", recording)
    s.send(b"\x13", 0.2)  # Ctrl+S
    wait_for(s, "decoding 00:2")
    s.pump(1.5)
    save(render(s), "transcribing.png")
    wait_for(s, "Finished")
    s.close()

    s = Session(root, "Sunset", recording)
    s.send(b"\x13", 0.2)
    wait_for(s, "Finished")
    s.pump(0.5)
    save(render(s), "done.png")
    s.close()

    s = Session(root, "Aurora", recording, path="/nonexistent")
    s.send(b"\x13", 0.6)
    save(render(s), "ffmpeg.png")
    s.close()


def themes(root, recording):
    tiles = []
    for name in THEMES:
        s = Session(root, name, recording)
        s.send(b"\t\t", 0.3)  # focus the model box so the highlight shows
        tiles.append(render(s, f"{name}"))
        s.close()
    tw = 640
    th = round(tiles[0].height * tw / tiles[0].width)
    cols = 3
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tw + (cols + 1) * 16, rows * th + (rows + 1) * 16), (24, 24, 32))
    for i, tile in enumerate(tiles):
        small = tile.resize((tw, th), Image.LANCZOS)
        sheet.paste(small, (16 + (i % cols) * (tw + 16), 16 + (i // cols) * (th + 16)))
    save(sheet, "themes.png")


def demo_gif(root, recording):
    frames, durations = [], []

    def grab(session, ms=250):
        frames.append(render(session))
        durations.append(ms)

    s = Session(root, "Aurora", recording)
    for _ in range(3):
        grab(s, 400)
    for _ in range(4):  # flip through themes
        s.send(b"\x14", 0.25)
        grab(s, 600)
    s.send(b"\x1bOP", 0.3)  # help
    for _ in range(4):
        grab(s, 450)
    s.send(b"\x1b", 0.3)
    s.send(b"\x0f", 0.4)  # browser
    grab(s, 900)
    s.send(b"\x1b", 0.3)
    s.hover_on("Start transcription")
    grab(s, 700)
    s.click_on("Start transcription")
    end = time.time() + 150
    while time.time() < end and "Finished" not in s.text():
        s.pump(0.45)
        grab(s, 220)
    for _ in range(8):
        s.pump(0.2)
        grab(s, 500)
    s.close()

    scale = 880 / frames[0].width
    size = (880, round(frames[0].height * scale))
    small = [f.resize(size, Image.LANCZOS).quantize(colors=96, method=Image.MEDIANCUT, dither=Image.NONE)
             for f in frames]
    OUT.mkdir(parents=True, exist_ok=True)
    small[0].save(OUT / "demo.gif", save_all=True, append_images=small[1:], duration=durations,
                  loop=0, optimize=True, disposal=1)
    print("wrote", OUT / "demo.gif", f"({len(frames)} frames)")


def main():
    if not Path(BINARY).exists():
        sys.exit(f"{BINARY} not found; build it first")
    # A fixed name, so paths on screen read the same every time.
    root = Path(tempfile.gettempdir()) / "whispry-demo"
    shutil.rmtree(root, ignore_errors=True)
    root.mkdir()
    try:
        recording = make_demo(root)
        stills(root, recording)
        themes(root, recording)
        demo_gif(root, recording)
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    main()
