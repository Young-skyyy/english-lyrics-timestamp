from __future__ import annotations

import logging
import math
import time
from typing import Any

from .models import AsrSegment, AsrWord

logger = logging.getLogger(__name__)


class WhisperXAcousticAligner:
    """使用 WhisperX 的 wav2vec2/CTC 对齐模型对已知文本进行声学对齐。"""

    def __init__(
        self,
        whisperx: Any,
        device: str = "cpu",
        align_model_name_or_path: str | None = None,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("WhisperX device 必须是 'cpu' 或 'cuda'")
        self.whisperx = whisperx
        self.device = device
        self.align_model_name_or_path = align_model_name_or_path
        self._align_model: Any | None = None
        self._align_metadata: Any | None = None

    def align(
        self,
        audio: Any,
        segments: list[dict[str, Any]],
        language: str = "en",
    ) -> list[AsrSegment]:
        """将官方歌词文本作为目标，在粗定位窗口内执行 WhisperX 强制对齐。"""
        if not segments:
            return []

        logger.info(
            "WhisperX 开始加载/复用声学对齐模型 language=%s device=%s input_segments=%d",
            language,
            self.device,
            len(segments),
        )
        align_model, metadata = self._get_align_model(language)
        started_at = time.perf_counter()
        cleaned = self._clean_windows(segments)
        logger.info(
            "WhisperX 对齐窗口已清理 input=%d output=%d first=%s last=%s",
            len(segments),
            len(cleaned),
            self._window_summary(cleaned[0]) if cleaned else None,
            self._window_summary(cleaned[-1]) if cleaned else None,
        )
        logger.info(
            "WhisperX 开始逐帧声学对齐 segments=%d window_span=%.2fs",
            len(cleaned),
            (cleaned[-1]["end"] - cleaned[0]["start"]) if cleaned else 0.0,
        )
        ctc_started_at = time.perf_counter()
        aligned = self._run_align(audio, cleaned, align_model, metadata)
        logger.info(
            "WhisperX 首次 CTC 对齐调用返回 elapsed=%.2fs raw_segments=%d",
            time.perf_counter() - ctc_started_at,
            len(aligned.get("segments", [])),
        )
        result = self._to_segments(aligned.get("segments", []))
        aligned_words = sum(len(segment.words) for segment in result)

        if result and aligned_words < max(1, len(cleaned) // 2):
            retry_windows = self._expand_windows(cleaned, 500)
            logger.warning(
                "WhisperX 首次对齐词数偏少 words=%d，扩大窗口重试 padding_ms=500",
                aligned_words,
            )
            retry_started_at = time.perf_counter()
            aligned = self._run_align(audio, retry_windows, align_model, metadata)
            logger.info(
                "WhisperX 扩大窗口 CTC 对齐调用返回 elapsed=%.2fs raw_segments=%d",
                time.perf_counter() - retry_started_at,
                len(aligned.get("segments", [])),
            )
            retry_result = self._to_segments(aligned.get("segments", []))
            retry_words = sum(len(segment.words) for segment in retry_result)
            if retry_words >= aligned_words:
                result = retry_result
                aligned_words = retry_words

        logger.info(
            "WhisperX 声学对齐完成 elapsed=%.2fs segments=%d words=%d",
            time.perf_counter() - started_at,
            len(result),
            aligned_words,
        )
        for index, segment in enumerate(result):
            logger.debug(
                "CTC 结果 segment=%d time=%d-%d words=%d text=%r",
                index,
                segment.start_ms,
                segment.end_ms,
                len(segment.words),
                segment.text,
            )
        return result

    def _run_align(
        self,
        audio: Any,
        segments: list[dict[str, Any]],
        align_model: Any,
        metadata: Any,
    ) -> dict[str, Any]:
        return self.whisperx.align(
            segments,
            align_model,
            metadata,
            audio,
            self.device,
            return_char_alignments=False,
        )

    @staticmethod
    def _clean_windows(segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cleaned: list[dict[str, Any]] = []
        previous_end = 0.0
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            start = max(previous_end, float(segment.get("start", previous_end)))
            end = max(start + 0.05, float(segment.get("end", start + 0.05)))
            cleaned.append({"text": text, "start": start, "end": end})
            previous_end = end
        return cleaned

    @staticmethod
    def _expand_windows(segments: list[dict[str, Any]], padding_s: float) -> list[dict[str, Any]]:
        return [
            {
                "text": segment["text"],
                "start": max(0.0, segment["start"] - padding_s / 1000),
                "end": segment["end"] + padding_s / 1000,
            }
            for segment in segments
        ]

    def _get_align_model(self, language: str) -> tuple[Any, Any]:
        if self._align_model is None:
            started_at = time.perf_counter()
            logger.info("WhisperX 正在加载 %s wav2vec2/CTC 模型", language)
            kwargs: dict[str, Any] = {"device": self.device}
            if self.align_model_name_or_path:
                kwargs["model_name"] = self.align_model_name_or_path
            self._align_model, self._align_metadata = self.whisperx.load_align_model(
                language_code=language,
                **kwargs,
            )
            logger.info(
                "WhisperX %s wav2vec2/CTC 模型加载完成 elapsed=%.2fs",
                language,
                time.perf_counter() - started_at,
            )
        else:
            logger.info("WhisperX 复用已加载的 %s wav2vec2/CTC 模型", language)
        return self._align_model, self._align_metadata

    @classmethod
    def _to_segments(cls, segments: list[dict[str, Any]]) -> list[AsrSegment]:
        results: list[AsrSegment] = []
        for segment in segments:
            text = str(segment.get("text", "")).strip()
            if not text:
                continue
            words: list[AsrWord] = []
            for word in segment.get("words", []) or []:
                word_text = str(word.get("word", word.get("text", ""))).strip()
                start = word.get("start")
                end = word.get("end")
                if not word_text or start is None or end is None:
                    continue
                words.append(
                    AsrWord(
                        text=word_text,
                        start_ms=round(float(start) * 1000),
                        end_ms=round(float(end) * 1000),
                        confidence=cls._confidence(word.get("score", word.get("probability", 0.0))),
                    )
                )
            results.append(
                AsrSegment(
                    text=text,
                    start_ms=round(float(segment.get("start", 0.0)) * 1000),
                    end_ms=round(float(segment.get("end", 0.0)) * 1000),
                    confidence=cls._segment_confidence(segment, words),
                    words=words,
                )
            )
        return results

    @staticmethod
    def _confidence(value: Any) -> float:
        try:
            return round(max(0.0, min(1.0, float(value))), 4)
        except (TypeError, ValueError):
            return 0.0

    @classmethod
    def _segment_confidence(cls, segment: dict[str, Any], words: list[AsrWord]) -> float:
        if words:
            return round(sum(word.confidence for word in words) / len(words), 4)
        score = segment.get("avg_logprob")
        if score is None:
            return 0.0
        try:
            return round(max(0.0, min(1.0, math.exp(float(score)))), 4)
        except (TypeError, ValueError, OverflowError):
            return 0.0


__all__ = ["WhisperXAcousticAligner"]
