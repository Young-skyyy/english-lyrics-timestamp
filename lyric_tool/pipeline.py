from __future__ import annotations

import logging
from dataclasses import asdict

from .aligner import LyricAligner
from .detector import IssueDetector
from .models import AlignmentResult, AsrSegment
from .parser import parse_lyrics, parse_lrc

logger = logging.getLogger(__name__)


class AlignmentPipeline:
    def __init__(self, aligner: LyricAligner | None = None, detector: IssueDetector | None = None):
        self.aligner = aligner or LyricAligner()
        self.detector = detector or IssueDetector()

    def run(self, lyrics_text: str, asr_segments: list[AsrSegment], imported_lrc: str | None = None) -> AlignmentResult:
        lyrics = parse_lyrics(lyrics_text)
        imported_lines = parse_lrc(imported_lrc) if imported_lrc else None
        lines, global_offset_ms = self.aligner.align(lyrics, asr_segments, imported_lines)
        issues = self.detector.detect(lines, imported_lines, global_offset_ms)
        matched = [
            line for line in lines
            if line.start_ms is not None and line.content_confidence > 0
            and line.match_method not in {"lrc_provisional", "asr_interpolated", "non_lexical_vocal"}
        ]
        coverage = len(matched) / len(lines) if lines else 0.0
        content_confidence = sum(line.content_confidence for line in matched) / len(matched) if matched else 0.0
        timing_confidence = sum(line.timing_confidence for line in matched) / len(matched) if matched else 0.0
        status = "needs_review" if issues else "approved"
        logger.info("对齐完成 status=%s coverage=%.4f content_confidence=%.4f timing_confidence=%.4f issue_count=%d", status, coverage, content_confidence, timing_confidence, len(issues))
        return AlignmentResult("line-v0.3", "line", status, global_offset_ms, round(content_confidence, 4), round(timing_confidence, 4), round(coverage, 4), lines, issues)
