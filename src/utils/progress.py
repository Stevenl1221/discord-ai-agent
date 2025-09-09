from __future__ import annotations

def bar(percent: int, width: int = 20, fill_char: str = "#", empty_char: str = "-") -> str:
    pct = max(0, min(100, int(percent)))
    filled = int((pct / 100.0) * width)
    return f"[{fill_char * filled}{empty_char * (width - filled)}] {pct}%"

