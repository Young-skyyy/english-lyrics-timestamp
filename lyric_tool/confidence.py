from __future__ import annotations


def calculate_content_confidence(
    text_match_score: float,
    asr_confidence: float,
    word_coverage: float = 0.0,
    sequence_score: float = 0.0,
) -> float:
    return round(
        0.45 * text_match_score
        + 0.30 * word_coverage
        + 0.15 * asr_confidence
        + 0.10 * sequence_score,
        4,
    )


def calculate_initial_timing_confidence(timing_score: float, sequence_score: float) -> float:
    return round(0.70 * timing_score + 0.30 * sequence_score, 4)
