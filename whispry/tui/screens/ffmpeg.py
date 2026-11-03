"""Modal shown when FFmpeg is missing: install steps, or the bundled build."""

from __future__ import annotations

from textual import on, work
from textual.app import ComposeResult
from textual.containers import Horizontal, Vertical
from textual.screen import ModalScreen
from textual.widgets import Button, Label, Static

from whispry.core.ffmpeg import BUNDLED_DOWNLOAD_SIZE, BUNDLED_PACKAGE


class FFmpegScreen(ModalScreen[bool]):
    """Dismisses with True once FFmpeg is available, False if the user gives up."""

    DEFAULT_CSS = """
    FFmpegScreen { align: center middle; }
    #dialog { width: 90; max-width: 95%; height: auto; border: thick $warning; background: $surface; padding: 1 2; }
    #dialog #title { text-style: bold; }
    #dialog #body { margin: 1 0; }
    #dialog #status { color: $warning; height: auto; }
    #dialog Horizontal { height: auto; }
    #dialog Button { margin-right: 1; }
    """

    BINDINGS = [("escape", "dismiss(False)", "Cancel")]

    def compose(self) -> ComposeResult:
        with Vertical(id="dialog"):
            yield Label("FFmpeg is needed to decode audio, but it was not found.", id="title")
            yield Static(
                "You can install FFmpeg on your system and add it to PATH "
                "(recommended if other programs use it too), or let Whispry "
                f"install a bundled build just for itself ({BUNDLED_PACKAGE}, "
                f"{BUNDLED_DOWNLOAD_SIZE}).",
                id="body",
            )
            yield Static("", id="status")
            with Horizontal():
                yield Button("Show install steps", id="steps")
                yield Button("Use bundled FFmpeg", variant="primary", id="bundled")
                yield Button("Check again", id="check")
                yield Button("Cancel", id="cancel")

    @property
    def services(self):
        return self.app.services

    @on(Button.Pressed, "#steps")
    def _show_steps(self) -> None:
        steps = self.services.ffmpeg_install_instructions()
        self.query_one("#body", Static).update(
            "\n".join(f"{n}. {step}" for n, step in enumerate(steps, 1))
            + "\n\nAfter installing, press Check again. If FFmpeg still isn't "
            "found, restart Whispry so it sees the updated PATH."
        )

    @on(Button.Pressed, "#check")
    def _check(self) -> None:
        if self.services.find_ffmpeg():
            self.dismiss(True)
        else:
            self.query_one("#status", Static).update("Still not found.")

    @on(Button.Pressed, "#cancel")
    def _cancel(self) -> None:
        self.dismiss(False)

    @on(Button.Pressed, "#bundled")
    def _install(self) -> None:
        for button in self.query(Button):
            button.disabled = True
        self.query_one("#status", Static).update(f"Installing {BUNDLED_PACKAGE}...")
        self._install_worker()

    @work(thread=True, exclusive=True)
    def _install_worker(self) -> None:
        result = self.services.install_bundled_ffmpeg()
        self.app.call_from_thread(self._installed, result)

    def _installed(self, result) -> None:
        if result.returncode == 0 and self.services.find_ffmpeg():
            self.dismiss(True)
            return
        for button in self.query(Button):
            button.disabled = False
        detail = (result.stderr or result.stdout or "").strip()[-600:]
        self.query_one("#status", Static).update(f"Installation failed:\n{detail}")
