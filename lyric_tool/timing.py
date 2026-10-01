from __future__ import annotations

import logging
from statistics import median

from .models import AsrSegment, AsrWord, ImportedLyricLine, LyricLine
from .matcher import align_lyric_lines_to_segments

logger = logging.getLogger(__name__)


def shift_segments(segments: list[AsrSegment], offset_ms: int) -> list[AsrSegment]:
    """将识别时间整体提前，保持文本和匹配顺序不变。"""
    if offset_ms <= 0:
        return segments

    shifted: list[AsrSegment] = []
    for segment in segments:
        start_ms = max(0, segment.start_ms - offset_ms)
        end_ms = max(start_ms, segment.end_ms - offset_ms)
        words = [
            AsrWord(
                text=word.text,
                start_ms=max(start_ms, word.start_ms - offset_ms),
                end_ms=max(start_ms, word.end_ms - offset_ms),
                confidence=word.confidence,
            )
            for word in segment.words
        ]
        shifted.append(
            AsrSegment(
                text=segment.text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=segment.confidence,
                words=words,
            )
        )
    return shifted


def correct_segments_piecewise(
    segments: list[AsrSegment],
    lyrics: list[LyricLine],
    imported_lines: list[ImportedLyricLine],
) -> list[AsrSegment]:
    """按歌词锚点估计分段偏移，并在线性插值下平滑校正时间。"""
    if not segments or not lyrics or not imported_lines:
        return segments

    matches = align_lyric_lines_to_segments(lyrics, segments)
    anchors: list[tuple[int, int]] = []
    used_segments: set[int] = set()
    last_segment_index = -1
    for lyric_index, segment_index, score in matches:
        if (
            lyric_index >= len(imported_lines)
            or segment_index >= len(segments)
            or score < 0.65
            or segment_index <= last_segment_index
            or segment_index in used_segments
        ):
            continue
        used_segments.add(segment_index)
        last_segment_index = segment_index
        anchors.append(
            (
                segments[segment_index].start_ms,
                segments[segment_index].start_ms - imported_lines[lyric_index].start_ms,
            )
        )
    if not anchors:
        logger.warning("分段时间校正跳过：没有足够可靠且单调的歌词锚点")
        return segments

    anchors.sort()

    def offset_at(timestamp_ms: int) -> float:
        if timestamp_ms <= anchors[0][0]:
            return float(anchors[0][1])
        if timestamp_ms >= anchors[-1][0]:
            return float(anchors[-1][1])
        for (left_time, left_offset), (right_time, right_offset) in zip(anchors, anchors[1:]):
            if timestamp_ms <= right_time:
                ratio = (timestamp_ms - left_time) / max(1, right_time - left_time)
                return left_offset + ratio * (right_offset - left_offset)
        return float(anchors[-1][1])

    def corrected(timestamp_ms: int, lower_bound: int = 0) -> int:
        return max(lower_bound, round(timestamp_ms - offset_at(timestamp_ms)))

    corrected_segments: list[AsrSegment] = []
    for segment in segments:
        start_ms = corrected(segment.start_ms)
        end_ms = corrected(segment.end_ms, start_ms)
        words: list[AsrWord] = []
        for word in segment.words:
            word_start = corrected(word.start_ms, start_ms)
            word_end = corrected(word.end_ms, word_start)
            words.append(AsrWord(word.text, word_start, word_end, word.confidence))
        corrected_segments.append(
            AsrSegment(segment.text, start_ms, end_ms, segment.confidence, words)
        )

    logger.info(
        "分段时间校正完成 anchors=%d first_offset_ms=%d last_offset_ms=%d",
        len(anchors),
        anchors[0][1],
        anchors[-1][1],
    )
    return corrected_segments


def calculate_timing_score(segment: AsrSegment, previous_end_ms: int | None) -> float:
    if previous_end_ms is None:
        return 1.0
    gap_ms = segment.start_ms - previous_end_ms
    if gap_ms < -300:
        return 0.25
    if gap_ms < 0:
        return 0.65
    if gap_ms <= 8_000:
        return 1.0
    return 0.60


def calculate_global_offset(deltas: list[int]) -> int:
    return round(median(deltas)) if deltas else 0


def calculate_timing_confidence(delta_ms: int, global_offset_ms: int) -> float:
    deviation_ms = abs(delta_ms - global_offset_ms)
    return round(max(0.0, 1.0 - min(deviation_ms / 1_000, 1.0)), 4)
