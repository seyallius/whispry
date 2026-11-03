"""Textual application shell."""

from __future__ import annotations

from pathlib import Path

from textual.app import App

from whispry.tui.screens.run import RunScreen
from whispry.tui.screens.setup import SetupScreen
from whispry.tui.services import Services


class WhispryApp(App):
    TITLE = "Whispry"
    SUB_TITLE = "Local speech-to-text"

    BINDINGS = [("ctrl+q", "quit", "Quit")]

    def __init__(self, services: Services | None = None, initial_file: Path | None = None):
        super().__init__()
        self.services = services or Services()
        self.initial_file = initial_file

    def on_mount(self) -> None:
        self.push_screen(SetupScreen(self.initial_file))

    def start_job(self, config) -> None:
        self.push_screen(RunScreen(config))
