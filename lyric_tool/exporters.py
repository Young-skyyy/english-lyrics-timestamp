from __future__ import annotations

from dataclasses import asdict
import json

from .models import AlignmentResult


def format_lrc(result: AlignmentResult) -> str:
    rows: list[str] = []
    for line in result.lines:
        if line.start_ms is None:
            continue
        # Round before splitting so 59.999 seconds becomes 01:00.00.
        centiseconds = (max(0, line.start_ms) + 5) // 10
        minutes, remainder = divmod(centiseconds, 6_000)
        seconds, fraction = divmod(remainder, 100)
        rows.append(f"[{minutes:02d}:{seconds:02d}.{fraction:02d}]{line.text}")
    return "\n".join(rows)


def format_json(result: AlignmentResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)
