from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class AsrWord:
    text: str
    start_ms: int
    end_ms: int
    confidence: float


@dataclass
class AsrSegment:
    text: str
    start_ms: int
    end_ms: int
    confidence: float
    words: list[AsrWord] = field(default_factory=list)


@dataclass
class LyricLine:
    index: int
    text: str
    normalized_text: str


@dataclass
class ImportedLyricLine:
    index: int
    text: str
    start_ms: int


@dataclass
class SyncLine:
    line_index: int
    text: str
    start_ms: int | None
    end_ms: int | None
    content_confidence: float
    timing_confidence: float
    match_method: str
    review_status: str
    text_match_score: float
    timing_score: float
    asr_score: float
    sequence_score: float


@dataclass
class ReviewIssue:
    issue_type: str
    severity: str
    status: str
    line_index: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


@dataclass
class AlignmentResult:
    algorithm_version: str
    sync_type: str
    status: str
    global_offset_ms: int
    content_confidence: float
    timing_confidence: float
    coverage_score: float
    lines: list[SyncLine]
    issues: list[ReviewIssue]

    @property
    def issue_count(self) -> int:
        return len(self.issues)
