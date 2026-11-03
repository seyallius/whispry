"""Modal for browsing to an audio or video file."""

from __future__ import annotations

from pathlib import Path

from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, DirectoryTree, Input, Label

MEDIA_SUFFIXES = {
    ".aac", ".aiff", ".avi", ".flac", ".m4a", ".mkv", ".mov", ".mp3", ".mp4",
    ".mpeg", ".oga", ".ogg", ".opus", ".wav", ".webm", ".wma", ".wmv",
}


class MediaTree(DirectoryTree):
    """Directory tree that shows folders and media files only."""

    def filter_paths(self, paths):
        return [
            path
            for path in paths
            if not path.name.startswith(".")
            and (path.is_dir() or path.suffix.lower() in MEDIA_SUFFIXES)
        ]


class FilePickerScreen(ModalScreen[Path | None]):
    DEFAULT_CSS = """
    FilePickerScreen { align: center middle; }
    #picker { width: 90%; height: 85%; border: thick $accent; background: $surface; padding: 1 2; }
    #picker MediaTree { height: 1fr; margin: 1 0; }
    #picker-buttons { height: auto; align-horizontal: right; }
    """

    BINDINGS = [("escape", "dismiss(None)", "Close")]

    def __init__(self, start: Path):
        super().__init__()
        self.start = start if start.is_dir() else Path.home()

    def compose(self) -> ComposeResult:
        with Vertical(id="picker"):
            yield Label("Choose an audio or video file. Type a folder and press Enter to jump there.")
            yield Input(str(self.start), id="folder")
            yield MediaTree(self.start, id="tree")
            with Horizontal(id="picker-buttons"):
                yield Button("Cancel", id="cancel")

    def on_input_submitted(self, event: Input.Submitted) -> None:
        folder = Path(event.value).expanduser()
        if folder.is_file():
            self.dismiss(folder)
        elif folder.is_dir():
            self.query_one(MediaTree).path = folder
        else:
            self.notify(f"Not a folder: {folder}", severity="error")

    def on_directory_tree_file_selected(self, event: DirectoryTree.FileSelected) -> None:
        self.dismiss(Path(event.path))

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(None)
