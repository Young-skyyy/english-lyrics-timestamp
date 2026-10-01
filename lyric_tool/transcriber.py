from __future__ import annotations

import logging
import math
import time
from difflib import SequenceMatcher
from pathlib import Path

from .models import AsrSegment, AsrWord
from .parser import parse_lyrics, normalize_text

logger = logging.getLogger(__name__)


class FasterWhisperTranscriber:
    """使用 faster-whisper 将音频转换为带时间戳的 ASR 片段。"""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
    ) -> None:
        try:
            from faster_whisper import WhisperModel
        except ImportError as exc:
            raise RuntimeError(
                "未安装 faster-whisper，请先运行: pip install faster-whisper"
            ) from exc

        logger.info(
            "加载 faster-whisper model=%s device=%s compute_type=%s",
            model_size,
            device,
            compute_type,
        )
        model_started_at = time.perf_counter()
        self.model = WhisperModel(
            model_size,
            device=device,
            compute_type=compute_type,
        )
        logger.info(
            "faster-whisper 模型加载完成 model=%s elapsed=%.2fs",
            model_size,
            time.perf_counter() - model_started_at,
        )

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None = None,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> list[AsrSegment]:
        logger.info("开始 ASR audio=%s language=%s", audio_path, language or "auto")
        asr_started_at = time.perf_counter()
        segments, info = self.model.transcribe(
            str(audio_path),
            language=language,
            beam_size=beam_size,
            vad_filter=vad_filter,
            word_timestamps=True,
        )

        logger.info(
            "ASR 推理迭代器已返回 elapsed=%.2fs duration=%.2fs detected_language=%s，开始消费识别结果",
            time.perf_counter() - asr_started_at,
            getattr(info, "duration", 0.0),
            getattr(info, "language", language or "unknown"),
        )
        results: list[AsrSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            confidence = self._segment_confidence(segment)
            words = [
                AsrWord(
                    text=word.text.strip(),
                    start_ms=round(word.start * 1_000),
                    end_ms=round(word.end * 1_000),
                    confidence=round(float(getattr(word, "probability", 0.0)), 4),
                )
                for word in (getattr(segment, "words", None) or [])
                if getattr(word, "text", "").strip()
            ]
            result = AsrSegment(
                text=text,
                start_ms=round(segment.start * 1_000),
                end_ms=round(segment.end * 1_000),
                confidence=confidence,
                words=words,
            )
            results.append(result)
            logger.info(
                "ASR segment text=%r time=[%d,%d] confidence=%.4f",
                result.text,
                result.start_ms,
                result.end_ms,
                result.confidence,
            )

        logger.info(
            "ASR 完成 language=%s duration=%.2fs segments=%d",
            getattr(info, "language", language or "unknown"),
            getattr(info, "duration", 0.0),
            len(results),
        )
        return results

    def refine_low_confidence(
        self,
        audio_path: str | Path,
        lyrics: str,
        segments: list[AsrSegment],
    ) -> list[AsrSegment]:
        """只重新识别低分片段，并用顺序合法的局部结果替换原结果。"""
        lyric_lines = parse_lyrics(lyrics)
        if not lyric_lines or not segments:
            return segments

        retry_ranges: list[tuple[int, int, str, float]] = []
        last_candidate_index = -1
        for lyric_position, lyric in enumerate(lyric_lines):
            expected_index = round(
                lyric_position * max(0, len(segments) - 1)
                / max(1, len(lyric_lines) - 1)
            )
            candidate_start = max(0, expected_index - 3, last_candidate_index + 1)
            candidate_end = min(len(segments), expected_index + 4)
            best_index = -1
            best_score = 0.0
            for index in range(candidate_start, candidate_end):
                segment = segments[index]
                score = SequenceMatcher(
                    None,
                    lyric.normalized_text,
                    normalize_text(segment.text),
                ).ratio()
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index < 0 or best_score >= 0.72:
                if best_index >= 0:
                    last_candidate_index = best_index
                continue
            last_candidate_index = best_index
            start_ms = max(0, segments[best_index].start_ms - 3_000)
            end_ms = segments[best_index].end_ms + 3_000
            retry_ranges.append((start_ms, end_ms, lyric.text, best_score))

        if not retry_ranges:
            return segments

        unique_ranges: list[tuple[int, int, str, float]] = []
        for retry_range in sorted(retry_ranges):
            start_ms, end_ms, prompt, old_score = retry_range
            if any(
                start_ms < existing_end and end_ms > existing_start
                for existing_start, existing_end, _prompt, _score in unique_ranges
            ):
                continue
            unique_ranges.append(retry_range)
            if len(unique_ranges) >= 20:
                break

        refined = list(segments)
        for start_ms, end_ms, prompt, old_score in unique_ranges:
            logger.info(
                "ASR 局部二次识别开始 range=%d-%d prompt=%r old_score=%.4f",
                start_ms,
                end_ms,
                prompt,
                old_score,
            )
            retry_segments, _info = self.model.transcribe(
                str(audio_path),
                language="en",
                beam_size=12,
                vad_filter=False,
                word_timestamps=True,
                condition_on_previous_text=False,
                initial_prompt=prompt,
                clip_timestamps=[start_ms / 1_000, end_ms / 1_000],
            )
            replacement = self._convert_segments(retry_segments)
            if not replacement:
                logger.info("ASR 局部二次识别无结果 range=%d-%d", start_ms, end_ms)
                continue
            new_score = max(
                SequenceMatcher(
                    None,
                    normalize_text(prompt),
                    normalize_text(segment.text),
                ).ratio()
                for segment in replacement
            )
            if new_score <= old_score:
                logger.info(
                    "ASR 局部二次识别结果未改善 range=%d-%d old_score=%.4f new_score=%.4f",
                    start_ms,
                    end_ms,
                    old_score,
                    new_score,
                )
                continue
            refined = [
                segment
                for segment in refined
                if segment.end_ms <= start_ms or segment.start_ms >= end_ms
            ]
            refined.extend(replacement)
            refined.sort(key=lambda segment: (segment.start_ms, segment.end_ms))
            logger.info(
                "ASR 局部二次识别完成 range=%d-%d replacement=%d old_score=%.4f new_score=%.4f",
                start_ms,
                end_ms,
                len(replacement),
                old_score,
                new_score,
            )
        return refined

    def _convert_segments(self, segments: object) -> list[AsrSegment]:
        results: list[AsrSegment] = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            words = [
                AsrWord(
                    text=word.text.strip(),
                    start_ms=round(word.start * 1_000),
                    end_ms=round(word.end * 1_000),
                    confidence=round(float(getattr(word, "probability", 0.0)), 4),
                )
                for word in (getattr(segment, "words", None) or [])
                if getattr(word, "text", "").strip()
            ]
            results.append(
                AsrSegment(
                    text=text,
                    start_ms=round(segment.start * 1_000),
                    end_ms=round(segment.end * 1_000),
                    confidence=self._segment_confidence(segment),
                    words=words,
                )
            )
        return results

    @staticmethod
    def _segment_confidence(segment: object) -> float:
        """将 Whisper 的 avg_logprob 转为 0~1 的审核排序分。"""
        avg_logprob = getattr(segment, "avg_logprob", None)
        if avg_logprob is None:
            return 0.0
        return round(max(0.0, min(1.0, math.exp(float(avg_logprob)))), 4)
