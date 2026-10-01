from __future__ import annotations

from .models import ImportedLyricLine, ReviewIssue, SyncLine


class IssueDetector:
    def detect(
        self,
        sync_lines: list[SyncLine],
        imported_lines: list[ImportedLyricLine] | None,
        global_offset_ms: int,
    ) -> list[ReviewIssue]:
        issues: list[ReviewIssue] = []
        if imported_lines:
            deltas = [
                line.start_ms - imported_lines[line.line_index].start_ms
                for line in sync_lines
                if line.start_ms is not None and line.line_index < len(imported_lines)
            ]
            stable_ratio = sum(delta * global_offset_ms >= 0 for delta in deltas) / len(deltas) if deltas and global_offset_ms else 0.0
            if abs(global_offset_ms) >= 250 and stable_ratio >= 0.70:
                issues.append(ReviewIssue("global_offset", "warning", "open", details={"offset_ms": global_offset_ms, "stable_ratio": round(stable_ratio, 4)}))
            for line in sync_lines:
                if line.start_ms is None or line.line_index >= len(imported_lines):
                    continue
                delta = line.start_ms - imported_lines[line.line_index].start_ms
                deviation = abs(delta - global_offset_ms)
                if deviation >= 300:
                    issues.append(ReviewIssue("local_offset", "warning", "open", line.line_index, {"original_start_ms": imported_lines[line.line_index].start_ms, "suggested_start_ms": line.start_ms, "delta_ms": delta, "deviation_ms": deviation}))

        for line in sync_lines:
            if line.start_ms is None:
                issues.append(ReviewIssue("missing_match", "blocking", "open", line.line_index, {"text": line.text}))
            elif line.text_match_score < 0.65:
                issues.append(ReviewIssue("text_mismatch", "warning", "open", line.line_index, {"text_match_score": line.text_match_score}))
            elif line.content_confidence < 0.75 or line.timing_confidence < 0.75:
                issues.append(ReviewIssue("low_confidence", "warning", "open", line.line_index, {"content_confidence": line.content_confidence, "timing_confidence": line.timing_confidence}))
        return issues
