from __future__ import annotations

import re

from .models import ImportedLyricLine, LyricLine


def normalize_text(text: str) -> str:
    text = text.casefold()
    return re.sub(r"[\W_]+", "", text, flags=re.UNICODE)


def _is_non_lyric_line(text: str) -> bool:
    normalized = text.strip().casefold()
    if re.fullmatch(
        r"\[(intro|verse|chorus|pre-chorus|bridge|hook|outro|interlude|break)(?:\s+\d+)?\]",
        normalized,
    ):
        return True
    return normalized.startswith("[") and normalized.endswith("]")


def parse_lyrics(raw_text: str) -> list[LyricLine]:
    lines: list[LyricLine] = []
    for text in raw_text.splitlines():
        text = text.strip()
        if not text or _is_non_lyric_line(text):
            continue
        lines.append(LyricLine(len(lines), text, normalize_text(text)))
    return lines


def parse_lrc(raw_text: str) -> list[ImportedLyricLine]:
    pattern = re.compile(r"\[(\d{1,3}):(\d{2})(?:[.:](\d{1,3}))?\](.*)")
    lines: list[ImportedLyricLine] = []
    for index, raw_line in enumerate(raw_text.splitlines()):
        match = pattern.fullmatch(raw_line.strip())
        if not match:
            continue
        minutes, seconds, fraction, text = match.groups()
        fraction_ms = int((fraction or "0").ljust(3, "0")[:3])
        start_ms = int(minutes) * 60_000 + int(seconds) * 1_000 + fraction_ms
        lyric_text = text.strip()
        if not lyric_text or _is_non_lyric_line(lyric_text):
            continue
        lines.append(ImportedLyricLine(len(lines), lyric_text, start_ms))
    return lines
