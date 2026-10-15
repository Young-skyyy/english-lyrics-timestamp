from __future__ import annotations

import logging
from difflib import SequenceMatcher

from dataclasses import dataclass
import re

from .models import AsrSegment, AsrWord, LyricLine
from .parser import normalize_text


@dataclass
class WordMatch:
    lyric_index: int
    asr_index: int
    score: float


_WORD_RE = re.compile(r"(?:[A-Za-z]+(?:['’-][A-Za-z]+)*)|(?:\d+(?:[.,:]\d+)*)")
_FUNCTION_WORDS = {
    "a", "an", "and", "as", "at", "be", "but", "by", "for", "from",
    "i", "if", "in", "is", "it", "of", "on", "or", "so", "that", "the",
    "to", "was", "we", "with", "you", "your",
}


def lyric_words(lyrics: list[LyricLine]) -> list[tuple[int, str]]:
    return [
        (line.index, word.casefold().replace("’", "'"))
        for line in lyrics
        for word in _WORD_RE.findall(line.text)
    ]


def _word_weight(word: str) -> float:
    normalized = word.casefold()
    if any(character.isdigit() for character in normalized):
        return 1.45
    if len(normalized) >= 8:
        return 1.25
    if normalized in _FUNCTION_WORDS or len(normalized) <= 2:
        return 0.65
    return 1.0


def _segment_words(segment: AsrSegment) -> list[AsrWord]:
    # Only measured word times belong in the acoustic word matching path.
    return [
        word for word in segment.words
        if word.text.strip() and 0 <= word.start_ms < word.end_ms
    ]


def asr_words(segments: list[AsrSegment]) -> list[tuple[int, AsrWord]]:
    return [
        (segment_index, word)
        for segment_index, segment in enumerate(segments)
        for word in _segment_words(segment)
    ]


def _approximate_word_score(expected: str, actual: str) -> float:
    expected = normalize_text(expected)
    actual = normalize_text(actual)
    if not expected or not actual:
        return 0.0
    if expected == actual:
        return 1.0
    score = SequenceMatcher(None, expected, actual).ratio()
    if expected.rstrip("s") == actual.rstrip("s"):
        score = max(score, 0.88)
    if expected.replace("wont", "want") == actual.replace("wont", "want"):
        score = max(score, 0.82)
    return score


def _window_score(expected: list[str], actual: list[str]) -> float:
    if not expected or not actual:
        return 0.0
    token_scores = [
        _approximate_word_score(expected[index], actual[index])
        for index in range(min(len(expected), len(actual)))
    ]
    coverage = len(token_scores) / len(expected)
    token_score = sum(token_scores) / len(token_scores)
    text_score = SequenceMatcher(
        None,
        "".join(normalize_text(word) for word in expected),
        "".join(normalize_text(word) for word in actual),
    ).ratio()
    return 0.55 * token_score + 0.30 * text_score + 0.15 * coverage


def align_lyric_lines_to_word_windows(
    lyrics: list[LyricLine],
    segments: list[AsrSegment],
) -> dict[int, tuple[list[int], float]]:
    """Globally match lyric words, allowing ASR insertions and missing words."""
    matches, scores = align_word_sequences(lyrics, segments)
    grouped: dict[int, list[int]] = {}
    for match in matches:
        grouped.setdefault(match.lyric_index, []).append(match.asr_index)
    return {
        index: (indexes, scores[index])
        for index, indexes in grouped.items()
        if scores[index] >= 0.55
    }


def align_word_sequences(
    lyrics: list[LyricLine], segments: list[AsrSegment]
) -> tuple[list[WordMatch], dict[int, float]]:
    """Globally align English lyric words and flattened ASR words in order."""
    expected = lyric_words(lyrics)
    observed = asr_words(segments)
    if not expected or not observed:
        return [], {}

    rows, cols = len(expected), len(observed)
    gap_lyric, gap_asr = -0.72, -0.28
    scores = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    moves = [["eq"] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        scores[row][0] = scores[row - 1][0] + gap_lyric
        moves[row][0] = "lyric"
    for col in range(1, cols + 1):
        scores[0][col] = scores[0][col - 1] + gap_asr
        moves[0][col] = "asr"

    for row in range(1, rows + 1):
        lyric_word = expected[row - 1][1]
        lyric_weight = _word_weight(lyric_word)
        for col in range(1, cols + 1):
            asr_word = observed[col - 1][1].text.casefold().replace("’", "'")
            ratio = SequenceMatcher(None, normalize_text(lyric_word), normalize_text(asr_word)).ratio()
            match_score = ratio * lyric_weight if ratio >= 0.55 else -0.80 * lyric_weight
            candidates = (
                (scores[row - 1][col - 1] + match_score, "eq"),
                (scores[row - 1][col] + gap_lyric * lyric_weight, "lyric"),
                (scores[row][col - 1] + gap_asr, "asr"),
            )
            scores[row][col], moves[row][col] = max(candidates, key=lambda item: item[0])

    matches: list[WordMatch] = []
    row, col = rows, cols
    while row or col:
        move = moves[row][col]
        if move == "eq":
            lyric_word = expected[row - 1][1]
            asr_word = observed[col - 1][1].text.casefold().replace("’", "'")
            ratio = SequenceMatcher(None, normalize_text(lyric_word), normalize_text(asr_word)).ratio()
            if ratio >= 0.55:
                matches.append(WordMatch(expected[row - 1][0], col - 1, ratio))
            row -= 1
            col -= 1
        elif move == "lyric":
            row -= 1
        else:
            col -= 1
    matches.reverse()
    line_scores: dict[int, list[float]] = {}
    for match in matches:
        line_scores.setdefault(match.lyric_index, []).append(match.score)
    expected_counts = {line.index: len(_WORD_RE.findall(line.text)) for line in lyrics}
    return matches, {
        index: round(
            (sum(values) / len(values)) * min(1.0, len(values) / max(1, expected_counts[index])),
            4,
        )
        for index, values in line_scores.items()
    }

logger = logging.getLogger(__name__)


def _segment_window_score(expected: list[str], actual_text: str) -> float:
    actual = _WORD_RE.findall(actual_text)
    if not expected or not actual:
        return 0.0
    actual_index = 0
    matched_scores: list[float] = []
    for expected_word in expected:
        best_index = None
        best_score = 0.0
        for index in range(actual_index, len(actual)):
            score = _approximate_word_score(expected_word, actual[index])
            if score > best_score:
                best_index = index
                best_score = score
        if best_index is not None and best_score >= 0.55:
            matched_scores.append(best_score)
            actual_index = best_index + 1
    coverage = len(matched_scores) / len(expected)
    token_score = sum(matched_scores) / len(matched_scores) if matched_scores else 0.0
    text_score = SequenceMatcher(
        None,
        normalize_text(" ".join(expected)),
        normalize_text(actual_text),
    ).ratio()
    length_penalty = min(1.0, len(expected) / max(len(actual), len(expected)))
    return 0.45 * token_score + 0.35 * coverage + 0.20 * text_score * length_penalty


def align_lyric_lines_to_segment_windows(
    lyrics: list[LyricLine],
    segments: list[AsrSegment],
    blocked_segments: set[int] | None = None,
    excluded_lyrics: set[int] | None = None,
    word_anchors: dict[int, tuple[int, int]] | None = None,
) -> dict[int, tuple[int, int, float]]:
    """按歌词顺序匹配连续的多个 ASR segment，并返回独占的 segment 区间。"""
    if not lyrics or not segments:
        return {}

    blocked = blocked_segments or set()
    excluded = excluded_lyrics or set()
    anchors = word_anchors or {}
    result: dict[int, tuple[int, int, float]] = {}
    cursor = 0
    for lyric in lyrics:
        if lyric.index in excluded or lyric.index in result:
            if lyric.index in anchors:
                cursor = max(cursor, anchors[lyric.index][1])
            continue
        expected = _WORD_RE.findall(lyric.text)
        if not expected:
            continue
        best: tuple[float, int, int] | None = None
        max_window_segments = min(
            12,
            max(2, len(expected) // 2 + 2),
        )
        next_anchor = min(
            (start for index, (start, _end) in anchors.items() if index > lyric.index),
            default=len(segments),
        )
        for start in range(cursor, next_anchor):
            if start in blocked:
                continue
            text_parts: list[str] = []
            for end in range(start, min(next_anchor, start + max_window_segments)):
                if end in blocked:
                    break
                text_parts.append(segments[end].text)
                actual = " ".join(text_parts)
                score = _segment_window_score(expected, actual)
                candidate = (score, start, end + 1)
                if best is None or candidate[0] > best[0] or (
                    candidate[0] == best[0]
                    and (candidate[2] - candidate[1]) < (best[2] - best[1])
                ) or (
                    candidate[0] == best[0]
                    and (candidate[2] - candidate[1]) == (best[2] - best[1])
                    and candidate[1] < best[1]
                ):
                    best = candidate
        if best is None or best[0] < 0.20:
            logger.debug(
                "连续 segment 窗口未匹配 line=%d text=%r cursor=%d score=%.4f",
                lyric.index,
                lyric.text,
                cursor,
                best[0] if best else 0.0,
            )
            continue
        score, start, end = best
        result[lyric.index] = (start, end, round(score, 4))
        blocked.update(range(start, end))
        cursor = end
        logger.debug(
            "连续 segment 窗口匹配 line=%d text=%r segments=%d-%d score=%.4f",
            lyric.index,
            lyric.text,
            start,
            end - 1,
            score,
        )
    return result


def align_lyric_lines_to_segments(
    lyrics: list[LyricLine],
    segments: list[AsrSegment],
) -> list[tuple[int, int, float]]:
    """用单调动态规划一次性匹配整首歌词和 ASR 片段。"""
    if not lyrics or not segments:
        return []

    rows, cols = len(lyrics), len(segments)
    gap_lyric, gap_segment = -0.65, -0.18
    scores = [[0.0] * (cols + 1) for _ in range(rows + 1)]
    moves = [["start"] * (cols + 1) for _ in range(rows + 1)]
    for row in range(1, rows + 1):
        scores[row][0] = scores[row - 1][0] + gap_lyric
        moves[row][0] = "lyric"
    for col in range(1, cols + 1):
        scores[0][col] = scores[0][col - 1] + gap_segment
        moves[0][col] = "segment"

    for row in range(1, rows + 1):
        lyric_text = lyrics[row - 1].normalized_text
        for col in range(1, cols + 1):
            text_score = SequenceMatcher(
                None,
                lyric_text,
                normalize_text(segments[col - 1].text),
            ).ratio()
            match_score = (text_score - 0.35) * _word_weight(lyric_text)
            candidates = (
                (scores[row - 1][col - 1] + match_score, "match"),
                (scores[row - 1][col] + gap_lyric, "lyric"),
                (scores[row][col - 1] + gap_segment, "segment"),
            )
            scores[row][col], moves[row][col] = max(
                candidates,
                key=lambda item: item[0],
            )

    matches: list[tuple[int, int, float]] = []
    used_segments: set[int] = set()
    last_segment_index = cols
    row, col = rows, cols
    while row or col:
        move = moves[row][col]
        if move == "match":
            lyric = lyrics[row - 1]
            segment = segments[col - 1]
            text_score = SequenceMatcher(
                None,
                lyric.normalized_text,
                normalize_text(segment.text),
            ).ratio()
            segment_index = col - 1
            if (
                text_score >= 0.20
                and segment_index not in used_segments
                and segment_index < last_segment_index
            ):
                matches.append((lyric.index, segment_index, text_score))
                used_segments.add(segment_index)
                last_segment_index = segment_index
            row -= 1
            col -= 1
        elif move == "lyric":
            row -= 1
        else:
            col -= 1
    matches.reverse()
    logger.info(
        "歌词行全局匹配完成 lyrics=%d segments=%d matched=%d",
        len(lyrics),
        len(segments),
        len(matches),
    )
    return matches


def best_segment_match(
    lyric: LyricLine,
    segments: list[AsrSegment],
    search_from: int,
) -> tuple[int | None, float]:
    best_index: int | None = None
    best_score = 0.0
    for index in range(search_from, len(segments)):
        score = SequenceMatcher(
            None, lyric.normalized_text, normalize_text(segments[index].text)
        ).ratio()
        logger.debug(
            "候选匹配 line=%d segment=%d score=%.4f",
            lyric.index,
            index,
            score,
        )
        if score > best_score:
            best_index, best_score = index, score
    logger.debug("最佳匹配 line=%d segment=%s score=%.4f", lyric.index, best_index, best_score)
    return best_index, best_score
