"""Terminal UI. Presentation only: all decisions are made by whispry.core."""


def main(initial_file=None) -> None:
    from whispry.tui.app import WhispryApp

    WhispryApp(initial_file=initial_file).run()
