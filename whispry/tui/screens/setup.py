"""First screen: choose the file and how to transcribe it."""

from __future__ import annotations

from pathlib import Path

from textual import on
from textual.app import ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.screen import Screen
from textual.widgets import (
    Button,
    Collapsible,
    Footer,
    Header,
    Input,
    Label,
    RadioButton,
    RadioSet,
    Select,
    Static,
)

from whispry.core.config import (
    DEFAULT_CHUNK_MINUTES,
    DEFAULT_MODEL,
    LANGUAGES,
    MODEL_REPOSITORIES,
    JobConfig,
)
from whispry.core.platform import AUTO, CPU, CUDA, DeviceUnavailable
from whispry.tui.screens.file_picker import FilePickerScreen

DEFAULT_COMPUTE = "default"


class SetupScreen(Screen):
    DEFAULT_CSS = """
    SetupScreen VerticalScroll { padding: 1 2; }
    SetupScreen .section { margin-top: 1; text-style: bold; }
    SetupScreen #system { color: $text-muted; }
    SetupScreen #file-row { height: auto; }
    SetupScreen #file-row Input { width: 1fr; }
    SetupScreen #gpu-reason { color: $warning; margin-left: 4; }
    SetupScreen #errors { color: $error; margin-top: 1; }
    SetupScreen #start { margin-top: 1; width: 100%; }
    SetupScreen RadioSet { width: 100%; }
    """

    BINDINGS = [
        ("ctrl+o", "browse", "Browse"),
        ("ctrl+s", "start", "Start"),
    ]

    def __init__(self, initial_file: Path | None = None):
        super().__init__()
        self.initial_file = initial_file

    @property
    def services(self):
        return self.app.services

    def compose(self) -> ComposeResult:
        system = self.services.detect_system()
        gpu = system.gpu

        yield Header()
        with VerticalScroll():
            yield Static(self._system_summary(system), id="system")

            yield Label("Audio or video file", classes="section")
            with Horizontal(id="file-row"):
                yield Input(
                    str(self.initial_file or ""),
                    placeholder="Path to a file, or press Browse",
                    id="file",
                )
                yield Button("Browse", id="browse")

            yield Label("Model", classes="section")
            yield Select(self._model_options(), value=DEFAULT_MODEL, allow_blank=False, id="model")

            yield Label("Language", classes="section")
            yield Select(
                [("Auto-detect", AUTO), *((name, code) for code, name in LANGUAGES.items())],
                value=AUTO,
                allow_blank=False,
                id="language",
            )

            yield Label("Device", classes="section")
            with RadioSet(id="device"):
                yield RadioButton(self._auto_label(), value=True, id=f"device-{AUTO}")
                yield RadioButton(gpu.label, id=f"device-{CUDA}", disabled=not gpu.available)
                yield RadioButton("CPU", id=f"device-{CPU}")
            if not gpu.available:
                yield Static(f"GPU unavailable: {gpu.reason}", id="gpu-reason")

            with Collapsible(title="Advanced", id="advanced"):
                yield Label("Compute type")
                yield Select(
                    self._compute_options(AUTO),
                    value=DEFAULT_COMPUTE,
                    allow_blank=False,
                    id="compute",
                )
                yield Label("Chunk length in minutes (lower uses less RAM)")
                yield Input(str(DEFAULT_CHUNK_MINUTES), type="number", id="chunk")

            yield Static("", id="errors")
            yield Button("Start transcription", variant="primary", id="start")
        yield Footer()

    # ----- labels built from core data -----

    @staticmethod
    def _system_summary(system) -> str:
        devices = ", ".join(d.label for d in system.available_devices)
        return f"{system.os_label} · {system.cpu_threads} CPU threads · Available: {devices}"

    def _auto_label(self) -> str:
        option, _ = self.services.resolve_device(AUTO, self.services.detect_system())
        return f"Auto (uses {option.label})"

    def _model_options(self):
        return [
            (f"{name}{' (downloaded)' if self.services.is_model_downloaded(name) else ''}", name)
            for name in MODEL_REPOSITORIES
        ]

    # ----- reactions -----

    def _selected_device(self) -> str:
        pressed = self.query_one("#device", RadioSet).pressed_button
        return pressed.id.removeprefix("device-") if pressed else AUTO

    def _compute_options(self, device: str):
        option, default = self.services.resolve_device(device, self.services.detect_system())
        return [(f"Default ({default})", DEFAULT_COMPUTE)] + [
            (name, name) for name in option.compute_types
        ]

    def _refresh_compute_types(self) -> None:
        select = self.query_one("#compute", Select)
        select.set_options(self._compute_options(self._selected_device()))
        select.value = DEFAULT_COMPUTE

    @on(RadioSet.Changed, "#device")
    def _device_changed(self) -> None:
        self._refresh_compute_types()

    @on(Button.Pressed, "#browse")
    def action_browse(self) -> None:
        current = Path(self.query_one("#file", Input).value).expanduser()
        start = current.parent if current.parent.is_dir() else Path.cwd()

        def chosen(path: Path | None) -> None:
            if path:
                self.query_one("#file", Input).value = str(path)

        self.app.push_screen(FilePickerScreen(start), chosen)

    @on(Button.Pressed, "#start")
    @on(Input.Submitted, "#file")
    def action_start(self) -> None:
        config, problems = self._build_config()
        self.query_one("#errors", Static).update("\n".join(problems))
        if not problems:
            self.app.start_job(config)

    def _build_config(self) -> tuple[JobConfig | None, list[str]]:
        raw_path = self.query_one("#file", Input).value.strip().strip('"')
        if not raw_path:
            return None, ["Choose a file to transcribe."]

        try:
            chunk = float(self.query_one("#chunk", Input).value)
        except ValueError:
            return None, ["Chunk length must be a number."]

        language = self.query_one("#language", Select).value
        compute = self.query_one("#compute", Select).value

        config = JobConfig(
            audio=Path(raw_path).expanduser(),
            model=self.query_one("#model", Select).value,
            language=None if language == AUTO else language,
            device=self._selected_device(),
            compute_type=None if compute == DEFAULT_COMPUTE else compute,
            chunk_minutes=chunk,
        )

        problems = self.services.validate_config(config)
        if not problems:
            try:
                self.services.resolve_device(
                    config.device, self.services.detect_system(), config.compute_type
                )
            except DeviceUnavailable as error:
                problems.append(str(error))
        return config, problems
