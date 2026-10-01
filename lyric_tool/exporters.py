from __future__ import annotations

from dataclasses import asdict
import json

from .models import AlignmentResult


def format_lrc(result: AlignmentResult) -> str:
    rows: list[str] = []
    for line in result.lines:
        if line.start_ms is None:
            continue
        minutes, remainder = divmod(line.start_ms, 60_000)
        rows.append(f"[{minutes:02d}:{remainder / 1_000:05.2f}]{line.text}")
    return "\n".join(rows)


def format_json(result: AlignmentResult) -> str:
    return json.dumps(asdict(result), ensure_ascii=False, indent=2)
