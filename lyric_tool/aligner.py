from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher

from .confidence import calculate_content_confidence, calculate_initial_timing_confidence
from .matcher import (
    align_lyric_lines_to_segment_windows,
    align_lyric_lines_to_segments,
    align_lyric_lines_to_word_windows,
    asr_words,
)
from .models import AsrSegment, ImportedLyricLine, LyricLine, SyncLine
from .parser import normalize_text

_NON_LEXICAL_RE = re.compile(
    r"^(?:m+|h+m*|uh+|ah+|aye+)$",
    re.IGNORECASE,
)
from .timing import (
    calculate_global_offset,
    calculate_timing_confidence,
    calculate_timing_score,
)

logger = logging.getLogger(__name__)


class LyricAligner:
    @staticmethod
    def _segment_bounds(segment: AsrSegment) -> tuple[int, int, str]:
        if segment.words:
            return segment.words[0].start_ms, segment.words[-1].end_ms, "asr_word"
        return segment.start_ms, segment.end_ms, "asr_segment"

    def _align_repeated_single_line(
        self,
        lyric: LyricLine,
        asr_segments: list[AsrSegment],
        imported_lines: list[ImportedLyricLine] | None,
    ) -> tuple[list[SyncLine], int]:
        candidates: list[tuple[AsrSegment, float]] = []
        for segment in asr_segments:
            score = SequenceMatcher(
                None,
                lyric.normalized_text,
                normalize_text(segment.text),
            ).ratio()
            if score >= 0.55:
                candidates.append((segment, score))

        sync_lines: list[SyncLine] = []
        deltas: list[int] = []
        previous_end_ms: int | None = None
        for occurrence_index, (segment, text_match_score) in enumerate(candidates):
            start_ms, end_ms, match_method = self._segment_bounds(segment)
            timing_segment = AsrSegment(
                segment.text,
                start_ms,
                end_ms,
                segment.confidence,
                segment.words,
            )
            timing_score = calculate_timing_score(timing_segment, previous_end_ms)
            sequence_score = 1.0
            asr_confidence = segment.confidence if segment.confidence > 0 else text_match_score
            word_coverage = min(1.0, len(segment.words) / max(1, len(lyric.text.split())))
            content_confidence = calculate_content_confidence(
                text_match_score,
                asr_confidence,
                word_coverage,
                sequence_score,
            )
            timing_confidence = calculate_initial_timing_confidence(
                timing_score,
                sequence_score,
            )
            # 单行歌词可能对应多次演唱，原始 LRC 的单个时间点不能作为逐次偏移锚点。
            sync_lines.append(
                SyncLine(
                    occurrence_index,
                    lyric.text,
                    start_ms,
                    end_ms,
                    content_confidence,
                    timing_confidence,
                    "repeated_" + match_method,
                    "auto_pass" if content_confidence >= 0.90 and timing_confidence >= 0.90 else "needs_review",
                    round(text_match_score, 4),
                    round(timing_score, 4),
                    round(asr_confidence, 4),
                    sequence_score,
                )
            )
            previous_end_ms = end_ms

        global_offset_ms = calculate_global_offset(deltas)
        logger.info(
            "重复歌词匹配完成 lyric=%d occurrences=%d global_offset_ms=%d",
            lyric.index,
            len(sync_lines),
            global_offset_ms,
        )
        return sync_lines, global_offset_ms

    def align(
        self,
        lyrics: list[LyricLine],
        asr_segments: list[AsrSegment],
        imported_lines: list[ImportedLyricLine] | None = None,
    ) -> tuple[list[SyncLine], int]:
        if len(lyrics) == 1:
            return self._align_repeated_single_line(lyrics[0], asr_segments, imported_lines)

        sync_lines: list[SyncLine] = []
        deltas: list[int] = []
        search_from = 0
        previous_end_ms: int | None = None
        word_windows = align_lyric_lines_to_word_windows(lyrics, asr_segments)
        word_observed = asr_words(asr_segments)
        line_word_ranges = {
            lyric_index: word_indexes
            for lyric_index, (word_indexes, _score) in word_windows.items()
        }
        word_scores = {
            lyric_index: score
            for lyric_index, (_word_indexes, score) in word_windows.items()
        }
        line_matches: dict[int, tuple[int, float]] = {}
        line_segment_ranges: dict[int, tuple[int, int]] = {}
        word_segment_indexes = {
            word_observed[index][0]
            for word_indexes in line_word_ranges.values()
            for index in word_indexes
        }
        used_segments = set(word_segment_indexes)
        excluded_lyrics = set(line_word_ranges)
        segment_windows = align_lyric_lines_to_segment_windows(
            lyrics,
            asr_segments,
            used_segments,
            excluded_lyrics,
        )
        last_segment_index = -1
        for lyric in lyrics:
            matched_words = line_word_ranges.get(lyric.index, [])
            if matched_words:
                last_segment_index = max(
                    last_segment_index,
                    max(word_observed[index][0] for index in matched_words),
                )
                continue
            window = segment_windows.get(lyric.index)
            if window is None:
                continue
            start_segment, end_segment, score = window
            if start_segment <= last_segment_index:
                continue
            line_matches[lyric.index] = (start_segment, score)
            line_segment_ranges[lyric.index] = (start_segment, end_segment)
            used_segments.update(range(start_segment, end_segment))
            last_segment_index = end_segment - 1
        logger.info(
            "ASR segment候选匹配完成 word_windows=%d segment_candidates=%d observed_words=%d",
            len(word_windows),
            len(line_matches),
            len(word_observed),
        )

        for position, lyric in enumerate(lyrics):
            matched_words = line_word_ranges.get(lyric.index, [])
            segment_match = line_matches.get(lyric.index)
            if matched_words and segment_match is not None:
                segment_match = None
            segment_match_score = segment_match[1] if segment_match else 0.0
            expected_word_count = max(1, len(lyric.text.split()))
            word_coverage = len(matched_words) / expected_word_count
            logger.debug(
                "低分匹配开始 line=%d text=%r word_matches=%d word_score=%.4f segment_index=%s segment_score=%.4f search_from=%d previous_end_ms=%s",
                position,
                lyric.text,
                len(matched_words),
                word_scores.get(lyric.index, 0.0),
                segment_match[0] if segment_match else None,
                segment_match_score,
                search_from,
                previous_end_ms,
            )
            if matched_words:
                first_word = word_observed[matched_words[0]][1]
                last_word = word_observed[matched_words[-1]][1]
                word_segment_index = word_observed[matched_words[0]][0]
                word_match_score = word_scores[lyric.index]
                use_segment_bounds = (
                    segment_match is not None
                    and (
                        word_coverage < 0.75
                        or segment_match[0] != word_segment_index
                    )
                )
                if use_segment_bounds:
                    segment_index = segment_match[0]
                    text_match_score = max(word_match_score, segment_match_score)
                    start_ms = end_ms = None
                    match_method = "asr_segment_dp_low_coverage"
                else:
                    segment_index = word_segment_index
                    text_match_score = word_match_score
                    start_ms, end_ms, match_method = first_word.start_ms, last_word.end_ms, "asr_word_dp"
                logger.debug(
                    "低分匹配候选 line=%d source=words word_score=%.4f segment_score=%.4f selected_score=%.4f word_time=%s-%s segment_index=%d",
                    position,
                    word_match_score,
                    segment_match_score,
                    text_match_score,
                    start_ms,
                    end_ms,
                    segment_index,
                )
            elif segment_match:
                segment_index, text_match_score = segment_match
                segment_range = line_segment_ranges.get(
                    lyric.index,
                    (segment_index, segment_index + 1),
                )
                start_ms = asr_segments[segment_range[0]].start_ms
                end_ms = asr_segments[segment_range[1] - 1].end_ms
                match_method = (
                    "asr_segment_window"
                    if segment_range[1] - segment_range[0] > 1
                    else "asr_segment_dp"
                )
            else:
                segment_index, text_match_score = (None, 0.0)
                start_ms = end_ms = None
                match_method = "asr_segment_dp"

            if segment_index is None or text_match_score < 0.20:
                rejection_reasons = []
                if segment_index is None:
                    rejection_reasons.append("no_selected_candidate")
                if text_match_score < 0.20:
                    rejection_reasons.append("below_threshold")
                if not matched_words and segment_match is None:
                    rejection_reasons.append("no_word_or_segment_match")
                logger.info(
                    "歌词行最终未匹配 line=%d text=%r word_indexes=%s word_score=%.4f segment_candidate=%s segment_score=%.4f reasons=%s",
                    position,
                    lyric.text,
                    matched_words,
                    word_scores.get(lyric.index, 0.0),
                    segment_match[0] if segment_match else None,
                    segment_match_score,
                    ",".join(rejection_reasons),
                )
                provisional_start = None
                provisional_end = None
                provisional_method = (
                    "non_lexical_vocal"
                    if _NON_LEXICAL_RE.fullmatch(normalize_text(lyric.text))
                    else "missing_match"
                )
                if imported_lines and position < len(imported_lines):
                    provisional_method = "lrc_provisional"
                    provisional_start = imported_lines[position].start_ms
                    provisional_end = (
                        imported_lines[position + 1].start_ms
                        if position + 1 < len(imported_lines)
                        else provisional_start + 5_000
                    )
                else:
                    previous_end = next(
                        (
                            line.end_ms
                            for line in reversed(sync_lines)
                            if line.end_ms is not None
                        ),
                        None,
                    )
                    next_start = None
                    for next_position in range(position + 1, len(lyrics)):
                        next_lyric = lyrics[next_position]
                        next_word_indexes = line_word_ranges.get(next_lyric.index, [])
                        if next_word_indexes:
                            next_start = word_observed[next_word_indexes[0]][1].start_ms
                            break
                        next_match = line_matches.get(next_lyric.index)
                        if next_match is not None and next_match[1] >= 0.20:
                            next_start = asr_segments[next_match[0]].start_ms
                            break
                    if previous_end is not None and next_start is not None:
                        if provisional_method == "missing_match":
                            provisional_method = "asr_interpolated"
                        provisional_start = previous_end
                        provisional_end = max(previous_end + 500, next_start)
                    elif previous_end is not None:
                        if provisional_method == "missing_match":
                            provisional_method = "asr_interpolated"
                        provisional_start = previous_end
                        provisional_end = previous_end + 5_000
                    elif next_start is not None:
                        if provisional_method == "missing_match":
                            provisional_method = "asr_interpolated"
                        provisional_start = max(0, next_start - 5_000)
                        provisional_end = next_start
                logger.warning(
                    "匹配失败 line=%d lyric=%r score=%.4f method=%s provisional=%s",
                    position,
                    lyric.text,
                    text_match_score,
                    match_method,
                    provisional_start is not None,
                )
                sync_lines.append(
                    SyncLine(
                        position,
                        lyric.text,
                        provisional_start,
                        provisional_end,
                        0.0,
                        0.0,
                        provisional_method,
                        "needs_review",
                        round(text_match_score, 4),
                        0.0,
                        0.0,
                        0.0,
                    )
                )
                continue

            segment = asr_segments[segment_index]
            if not matched_words or start_ms is None or end_ms is None:
                start_ms, end_ms, match_method = self._segment_bounds(segment)
                logger.debug(
                    "低分匹配边界 line=%d source=%s segment=%d segment_time=%d-%d",
                    position,
                    match_method,
                    segment_index,
                    start_ms,
                    end_ms,
                )
            timing_segment = AsrSegment(
                text=segment.text,
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=segment.confidence,
                words=segment.words,
            )
            timing_score = calculate_timing_score(timing_segment, previous_end_ms)
            sequence_score = 1.0 if segment_index >= search_from else 0.25
            asr_confidence = segment.confidence if segment.confidence > 0 else text_match_score
            content_confidence = calculate_content_confidence(
                text_match_score,
                asr_confidence,
                min(1.0, word_coverage),
                sequence_score,
            )
            timing_confidence = calculate_initial_timing_confidence(timing_score, sequence_score)
            logger.debug(
                "低分匹配评分 line=%d text_score=%.4f asr_confidence=%.4f content=%.4f timing_score=%.4f sequence=%.4f timing=%.4f previous_end_ms=%s",
                position,
                text_match_score,
                asr_confidence,
                content_confidence,
                timing_score,
                sequence_score,
                timing_confidence,
                previous_end_ms,
            )
            if imported_lines and position < len(imported_lines):
                deltas.append(start_ms - imported_lines[position].start_ms)
                logger.debug(
                    "低分匹配偏移 line=%d observed_start_ms=%d imported_start_ms=%d delta_ms=%d",
                    position,
                    start_ms,
                    imported_lines[position].start_ms,
                    start_ms - imported_lines[position].start_ms,
                )

            review_status = (
                "auto_pass"
                if content_confidence >= 0.90 and timing_confidence >= 0.90
                else "needs_review"
            )
            logger.debug(
                "歌词行对齐 line=%d method=%s segment=%s words=%d time=%s-%s text_score=%.4f timing_score=%.4f content=%.4f timing=%.4f status=%s",
                lyric.index,
                match_method,
                segment_index,
                len(matched_words),
                start_ms,
                end_ms,
                text_match_score,
                timing_score,
                content_confidence,
                timing_confidence,
                review_status,
            )
            sync_lines.append(SyncLine(position, lyric.text, start_ms, end_ms, content_confidence, timing_confidence, match_method, review_status, round(text_match_score, 4), round(timing_score, 4), round(asr_confidence, 4), round(sequence_score, 4)))
            previous_end_ms = end_ms
            search_from = segment_index + 1

        global_offset_ms = calculate_global_offset(deltas)
        matched_count = sum(line.start_ms is not None for line in sync_lines)
        logger.info(
            "歌词对齐完成 matched=%d/%d global_offset_ms=%d deltas=%d",
            matched_count,
            len(sync_lines),
            global_offset_ms,
            len(deltas),
        )
        if imported_lines:
            for line in sync_lines:
                if line.start_ms is None or line.line_index >= len(imported_lines):
                    continue
                delta_ms = line.start_ms - imported_lines[line.line_index].start_ms
                line.timing_confidence = calculate_timing_confidence(delta_ms, global_offset_ms)
                line.review_status = "auto_pass" if line.content_confidence >= 0.90 and line.timing_confidence >= 0.90 else "needs_review"
        return sync_lines, global_offset_ms
