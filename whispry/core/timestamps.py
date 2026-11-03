"""Timestamp formatting."""


def format_timestamp(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)

    if hours:
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    return f"{minutes:02d}:{seconds:02d}"
