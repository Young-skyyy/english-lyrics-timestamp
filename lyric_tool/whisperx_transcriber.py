from __future__ import annotations

import logging
import math
import os
import time
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

from .parser import normalize_text

from .acoustic_aligner import WhisperXAcousticAligner
from .models import AsrSegment
from .parser import parse_lyrics
from .matcher import align_lyric_lines_to_segments

logger = logging.getLogger(__name__)


class WhisperXTranscriber:
    """可选的 WhisperX 英文转录器，并使用 wav2vec2 进行词级强制对齐。"""

    def __init__(
        self,
        model_name_or_path: str = "base.en",
        device: str = "cpu",
        compute_type: str | None = None,
        align_model_name_or_path: str | None = None,
        download_root: str | Path | None = None,
        batch_size: int = 16,
    ) -> None:
        whisperx = self._load_whisperx()
        if device not in {"cpu", "cuda"}:
            raise ValueError("WhisperX device 必须是 'cpu' 或 'cuda'")
        if batch_size < 1:
            raise ValueError("batch_size 必须大于 0")

        self.whisperx = whisperx
        self.device = device
        self.compute_type = compute_type or ("int8" if device == "cpu" else "float16")
        from pyannote.core import Segment
        from whisperx.vads.vad import Vad

        class FullAudioVad(Vad):
            def __init__(self) -> None:
                super().__init__(0.5)

            @staticmethod
            def preprocess_audio(audio: Any) -> Any:
                return audio

            def __call__(self, audio: Any, **kwargs: Any) -> list[Any]:
                duration = audio["waveform"].shape[-1] / 16_000
                return [Segment(0, duration)]

        self._vad_model = FullAudioVad()
        self.batch_size = batch_size
        self.align_model_name_or_path = align_model_name_or_path
        self.acoustic_aligner = WhisperXAcousticAligner(
            whisperx,
            device=device,
            align_model_name_or_path=align_model_name_or_path,
        )

        kwargs: dict[str, Any] = {
            "device": device,
            "compute_type": self.compute_type,
            "language": "en",
            "vad_model": self._vad_model,
        }
        if download_root is not None:
            kwargs["download_root"] = str(download_root)
        logger.info(
            "加载 WhisperX model=%s device=%s compute_type=%s batch_size=%d kwargs=%s",
            model_name_or_path,
            device,
            self.compute_type,
            batch_size,
            {key: value for key, value in kwargs.items() if key != "download_root"},
        )
        logger.info("WhisperX 主模型加载开始：可能包含模型下载、Tokenizer 初始化和 CPU 图构建")
        model_started_at = time.perf_counter()
        self.model = whisperx.load_model(model_name_or_path, **kwargs)
        logger.info("WhisperX 主模型加载完成 elapsed=%.2fs", time.perf_counter() - model_started_at)
        # region debug-point vad-runtime: report the active VAD implementation without changing behavior
        logger.info(
            "[DEBUG] WhisperX VAD runtime class=%s module=%s",
            type(getattr(self.model, "vad_model", None)).__name__,
            type(getattr(self.model, "vad_model", None)).__module__,
        )
        # endregion

    @staticmethod
    def _load_whisperx() -> Any:
        ffmpeg_bin = os.environ.get("FFMPEG_BIN")
        if ffmpeg_bin and hasattr(os, "add_dll_directory"):
            os.add_dll_directory(ffmpeg_bin)
            logger.info("已加入 FFmpeg DLL 目录: %s", ffmpeg_bin)
        try:
            import whisperx
        except ImportError as exc:
            raise RuntimeError(
                "WhisperX 未安装。英文转录和 wav2vec2 强制对齐需要可选依赖，"
                "请先运行: pip install whisperx"
            ) from exc
        return whisperx

    def transcribe(
        self,
        audio_path: str | Path,
        beam_size: int = 5,
        vad_filter: bool = True,
    ) -> list[AsrSegment]:
        """转录英文音频，并返回统一的 AsrSegment/AsrWord 结构。"""
        if beam_size < 1:
            raise ValueError("beam_size 必须大于 0")
        logger.info("WhisperX 开始读取音频 path=%s", audio_path)
        audio = self.whisperx.load_audio(str(audio_path))
        logger.info("WhisperX 开始模型推理 batch_size=%d beam_size=%d", self.batch_size, beam_size)
        transcription_started_at = time.perf_counter()
        options = {"language": "en", "batch_size": self.batch_size, "beam_size": beam_size}
        result = self.model.transcribe(audio, **options)
        logger.info("WhisperX 主模型推理完成 elapsed=%.2fs", time.perf_counter() - transcription_started_at)
        segments = result.get("segments", [])
        language = result.get("language", "en")
        return self._aligned_segments(audio, segments, language)

    def transcribe_official_lyrics(
        self,
        audio_path: str | Path,
        lyrics_text: str,
        beam_size: int = 8,
    ) -> list[AsrSegment]:
        """先用 ASR 找到歌词行的大致位置，再用官方歌词执行 CTC 对齐。"""
        lyrics = parse_lyrics(lyrics_text)
        if not lyrics:
            return []

        logger.info("WhisperX 开始读取 Rap 音频 path=%s", audio_path)
        audio_load_started_at = time.perf_counter()
        audio = self.whisperx.load_audio(str(audio_path))
        logger.info(
            "WhisperX Rap 音频读取完成 elapsed=%.2fs audio_samples=%d audio_seconds=%.2f",
            time.perf_counter() - audio_load_started_at,
            len(audio),
            len(audio) / 16_000,
        )
        started_at = time.perf_counter()
        logger.info(
            "WhisperX Rap 开始粗定位 requested_beam_size=%d api_beam_size=default batch_size=%d vad_filter=True audio_samples=%d audio_seconds=%.2f",
            beam_size,
            self.batch_size,
            len(audio),
            len(audio) / 16_000,
        )
        rough_inference_started_at = time.perf_counter()
        rough_result = self.model.transcribe(
            audio,
            language="en",
            batch_size=self.batch_size,
        )
        logger.info(
            "WhisperX Rap 粗识别调用返回 elapsed=%.2fs result_keys=%s",
            time.perf_counter() - rough_inference_started_at,
            sorted(rough_result.keys()),
        )
        rough_segments = [
            {
                "text": str(item.get("text", "")).strip(),
                "start": float(item.get("start", 0.0)),
                "end": float(item.get("end", 0.0)),
            }
            for item in rough_result.get("segments", [])
            if str(item.get("text", "")).strip()
        ]
        for index, segment in enumerate(rough_segments):
            logger.debug(
                "Rap 粗识别 segment=%d time=%.3f-%.3f text=%r",
                index,
                segment["start"],
                segment["end"],
                segment["text"],
            )
        logger.info(
            "WhisperX Rap 粗定位完成 elapsed=%.2fs segments=%d",
            time.perf_counter() - started_at,
            len(rough_segments),
        )

        rough_asr = [
            AsrSegment(
                text=item["text"],
                start_ms=round(item["start"] * 1000),
                end_ms=round(item["end"] * 1000),
                confidence=0.0,
            )
            for item in rough_segments
        ]
        official_segments: list[dict[str, Any]] = []
        path_matches: list[tuple[int, int, float]] = []
        if len(lyrics) == 1:
            lyric_text = normalize_text(lyrics[0].text)
            for index, rough in enumerate(rough_segments):
                score = SequenceMatcher(
                    None,
                    lyric_text,
                    normalize_text(rough["text"]),
                ).ratio()
                if score < 0.35:
                    continue
                official_segments.append(
                    {
                        "text": lyrics[0].text,
                        "start": rough["start"],
                        "end": rough["end"],
                    }
                )
                logger.debug(
                    "Rap 单句按时间顺序保留候选 segment=%d score=%.4f time=%.3f-%.3f",
                    index,
                    score,
                    rough["start"],
                    rough["end"],
                )
        else:
            path_matches = align_lyric_lines_to_segments(lyrics, rough_asr)

        for lyric_index, segment_index, score in path_matches if len(lyrics) > 1 else []:
            if score < 0.20:
                logger.warning(
                    "WhisperX Rap 全局定位不确定 line=%d segment=%d score=%.4f",
                    lyric_index,
                    segment_index,
                    score,
                )
                continue
            rough = rough_segments[segment_index]
            official_segments.append(
                {
                    "text": next(line.text for line in lyrics if line.index == lyric_index),
                    "start": rough["start"],
                    "end": rough["end"],
                }
            )
            logger.debug(
                "Rap 官方歌词窗口 line=%d segment=%d score=%.4f time=%.3f-%.3f",
                lyric_index,
                segment_index,
                score,
                rough["start"],
                rough["end"],
            )

        logger.info(
            "WhisperX Rap 官方歌词窗口构造完成 matched=%d total=%d",
            len(official_segments),
            len(lyrics),
        )
        return self._aligned_segments(audio, official_segments, "en")

    @staticmethod
    def _match_rap_segment(
        lyric: Any,
        segments: list[AsrSegment],
        search_from: int,
    ) -> tuple[int | None, float]:
        if search_from >= len(segments):
            return None, 0.0

        lyric_text = normalize_text(lyric.text)
        local_end = min(len(segments), search_from + 12)
        candidates = list(range(search_from, local_end))
        best_index, best_score = WhisperXTranscriber._score_rap_candidates(
            lyric_text,
            segments,
            candidates,
            search_from,
        )
        if best_index is not None and best_score >= 0.20:
            return best_index, best_score

        anchor_ms = segments[search_from].start_ms
        extended = [
            index
            for index in range(local_end, len(segments))
            if segments[index].start_ms - anchor_ms <= 20_000
        ]
        if extended:
            fallback_index, fallback_score = WhisperXTranscriber._score_rap_candidates(
                lyric_text,
                segments,
                extended,
                search_from,
                jump_penalty=True,
            )
            if fallback_score > best_score:
                return fallback_index, fallback_score
        return best_index, best_score

    @staticmethod
    def _score_rap_candidates(
        lyric_text: str,
        segments: list[AsrSegment],
        candidates: list[int],
        search_from: int,
        jump_penalty: bool = False,
    ) -> tuple[int | None, float]:
        best_index: int | None = None
        best_score = 0.0
        anchor_ms = segments[search_from].start_ms
        for index in candidates:
            text_score = SequenceMatcher(
                None,
                lyric_text,
                normalize_text(segments[index].text),
            ).ratio()
            distance_s = max(0.0, (segments[index].start_ms - anchor_ms) / 1000)
            penalty = min(distance_s / 20.0, 1.0) * 0.35 if jump_penalty else 0.0
            score = text_score - penalty
            logger.debug(
                "Rap 粗定位候选 segment=%d text_score=%.4f penalty=%.4f score=%.4f distance_s=%.2f",
                index,
                text_score,
                penalty,
                score,
                distance_s,
            )
            if score > best_score:
                best_index, best_score = index, score
        return best_index, best_score

    def _aligned_segments(self, audio: Any, segments: list[dict[str, Any]], language: str) -> list[AsrSegment]:
        return self.acoustic_aligner.align(audio, segments, language)



WhisperXTranscriberEnglish = WhisperXTranscriber

__all__ = ["WhisperXTranscriber", "WhisperXTranscriberEnglish"]
