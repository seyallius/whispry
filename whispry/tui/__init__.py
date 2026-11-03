"""Terminal UI. Presentation only: all decisions are made by whispry.core."""


def main(initial_file=None) -> None:
    from whispry.console import ConsoleRenderer
    from whispry.core import run_job
    from whispry.tui.app import WhispryApp

    config = WhispryApp(initial_file=initial_file).run()
    if config is not None:
        run_job(config, ConsoleRenderer())
