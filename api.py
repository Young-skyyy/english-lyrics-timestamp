from __future__ import annotations

import asyncio
import logging
import os
import time
import tempfile
import uuid
from dataclasses import asdict
from pathlib import Path
from threading import Lock

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from lyric_tool import AlignmentPipeline, FasterWhisperTranscriber, WhisperXTranscriber
from lyric_tool.exporters import format_lrc
from lyric_tool.models import AsrSegment
from lyric_tool.parser import parse_lyrics, parse_lrc
from lyric_tool.separator import DemucsSeparator
from lyric_tool.isolated_transcriber import run_transcription
from lyric_tool.lyric_import import MAX_LYRIC_BYTES, import_lyrics_file

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")

BASE_DIR = Path(__file__).resolve().parent
WEB_DIR = BASE_DIR / "web"
MAX_AUDIO_BYTES = 200 * 1024 * 1024
MODEL_LOAD_TIMEOUT_S = max(1, int(os.getenv("LYRICFLOW_MODEL_LOAD_TIMEOUT_S", "600")))
TRANSCRIPTION_TIMEOUT_S = max(1, int(os.getenv("LYRICFLOW_TRANSCRIPTION_TIMEOUT_S", "900")))
ALIGNMENT_TIMEOUT_S = max(1, int(os.getenv("LYRICFLOW_ALIGNMENT_TIMEOUT_S", "300")))
SEPARATION_TIMEOUT_S = max(1, int(os.getenv("LYRICFLOW_SEPARATION_TIMEOUT_S", "1800")))

app = FastAPI(title="LyricFlow API", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")

_transcriber: FasterWhisperTranscriber | None = None
_rap_transcriber: WhisperXTranscriber | None = None
_transcriber_lock = Lock()
_rap_transcriber_lock = Lock()
_transcriber_load_lock = asyncio.Lock()
_separator = DemucsSeparator(model_name="htdemucs", device="cpu")
_task_status: dict[str, dict[str, str]] = {}


def set_task_status(task_id: str, stage: str, message: str) -> None:
    _task_status[task_id] = {"stage": stage, "message": message}


@app.get("/", response_class=FileResponse)
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "lyricflow"}


@app.post("/api/lyrics/import")
async def import_lyrics(file: UploadFile = File(...)) -> dict[str, object]:
    data = await file.read(MAX_LYRIC_BYTES + 1)
    try:
        return import_lyrics_file(file.filename or '', data)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@app.get("/api/status/{task_id}")
def task_status(task_id: str) -> dict[str, str]:
    return _task_status.get(
        task_id,
        {"stage": "queued", "message": "任务已排队"},
    )


def build_fallback_asr(
    lyrics: str,
    imported_lrc: str | None,
    duration_ms: int,
) -> list[AsrSegment]:
    if imported_lrc:
        imported_lines = parse_lrc(imported_lrc)
        return [
            AsrSegment(
                text=line.text,
                start_ms=line.start_ms,
                end_ms=(
                    imported_lines[index + 1].start_ms
                    if index + 1 < len(imported_lines)
                    else min(duration_ms, line.start_ms + 5_000)
                ),
                confidence=0.5,
            )
            for index, line in enumerate(imported_lines)
        ]

    lyric_lines = parse_lyrics(lyrics)
    if not lyric_lines:
        return []
    interval_ms = max(1, duration_ms // len(lyric_lines))
    return [
        AsrSegment(
            text=line.text,
            start_ms=index * interval_ms,
            end_ms=min(duration_ms, (index + 1) * interval_ms),
            confidence=0.3,
        )
        for index, line in enumerate(lyric_lines)
    ]


async def save_upload(upload: UploadFile, destination: Path) -> int:
    total = 0
    with destination.open("wb") as output:
        while chunk := await upload.read(1024 * 1024):
            total += len(chunk)
            if total > MAX_AUDIO_BYTES:
                raise HTTPException(status_code=413, detail="音频文件不能超过 200 MB")
            output.write(chunk)
    return total


@app.post("/api/align")
async def align(
    audio: UploadFile = File(...),
    lyrics: str = Form(...),
    imported_lrc: str | None = Form(None),
    language: str | None = Form(None),
    mode: str = Form("fast"),
    duration_ms: int = Form(0),
    task_id: str | None = Form(None),
) -> dict[str, object]:
    task_id = task_id or uuid.uuid4().hex
    set_task_status(task_id, "queued", "任务已接收")
    if not audio.filename:
        raise HTTPException(status_code=400, detail="缺少音频文件名")
    if not lyrics.strip():
        raise HTTPException(status_code=400, detail="歌词不能为空")
    if mode not in {"fast", "rap", "quality"}:
        raise HTTPException(status_code=400, detail="mode 只能是 fast、rap 或 quality")

    suffix = Path(audio.filename).suffix.lower() or ".audio"
    with tempfile.TemporaryDirectory(prefix="lyricflow-") as temp_dir:
        audio_path = Path(temp_dir) / f"input{suffix}"
        upload_started_at = time.perf_counter()
        uploaded_bytes = await save_upload(audio, audio_path)
        logger.info(
            "[TASK] 音频保存完成 filename=%s bytes=%d elapsed=%.2fs",
            audio.filename,
            uploaded_bytes,
            time.perf_counter() - upload_started_at,
        )
        logger.info(
            "收到对齐任务 filename=%s size=%d duration_ms=%d",
            audio.filename,
            audio_path.stat().st_size,
            duration_ms,
        )

        try:
            request_started_at = time.perf_counter()
            set_task_status(task_id, "preparing", "正在准备音频")
            logger.info("[TASK] 开始处理 mode=%s filename=%s", mode, audio.filename)
            transcription_path = audio_path
            if mode == "quality":
                separated_dir = Path(temp_dir) / "separated"
                separation_started_at = time.perf_counter()
                logger.info("[TASK] 开始 Demucs 人声分离")
                transcription_path = await asyncio.wait_for(
                    asyncio.to_thread(
                        _separator.separate,
                        audio_path,
                        separated_dir,
                    ),
                    timeout=SEPARATION_TIMEOUT_S,
                )
                logger.info(
                    "[TASK] Demucs 人声分离完成 elapsed=%.2fs transcription_path=%s",
                    time.perf_counter() - separation_started_at,
                    transcription_path,
                )
            logger.info(
                "选择转录音频 path=%s mode=%s",
                transcription_path,
                mode,
            )
            clean_lyrics = (
                "\n".join(line.text for line in parse_lrc(imported_lrc))
                if imported_lrc
                else lyrics
            )
            logger.info(
                "歌词文本清洗完成 source=%s lines=%d chars=%d imported_lrc=%s",
                "lrc" if imported_lrc else "text",
                len(parse_lyrics(clean_lyrics)),
                len(clean_lyrics),
                bool(imported_lrc),
            )

            if mode == "rap":
                set_task_status(task_id, "loading_model", "正在加载 Rap 高精度模型，首次使用可能较慢")
            else:
                set_task_status(task_id, "loading_model", "正在加载识别模型")
            model_load_started_at = time.perf_counter()
            logger.info(
                "[TASK] 开始获取隔离识别器 mode=%s",
                mode,
            )
            set_task_status(task_id, "transcribing", "正在加载模型并进行语音识别")
            transcription_started_at = time.perf_counter()
            logger.info(
                "[TASK] 开始隔离语音识别 mode=%s audio=%s lyrics_chars=%d",
                mode,
                transcription_path,
                len(clean_lyrics),
            )
            try:
                asr_segments = await asyncio.to_thread(
                    run_transcription,
                    mode,
                    transcription_path,
                    None if language in {None, "", "auto"} else language,
                    clean_lyrics,
                    MODEL_LOAD_TIMEOUT_S,
                    TRANSCRIPTION_TIMEOUT_S,
                )
            except TimeoutError as exc:
                logger.exception(
                    "[TASK] 语音识别超时或模型加载超时 mode=%s elapsed=%.2fs",
                    mode,
                    time.perf_counter() - transcription_started_at,
                )
                raise RuntimeError(str(exc)) from exc
            logger.info(
                "[TASK] 隔离识别器获取及识别完成 mode=%s load_elapsed=%.2fs elapsed=%.2fs",
                mode,
                time.perf_counter() - model_load_started_at,
                time.perf_counter() - transcription_started_at,
            )
            logger.info(
                "[TASK] 语音识别完成 mode=%s elapsed=%.2fs segments=%d",
                mode,
                time.perf_counter() - transcription_started_at,
                len(asr_segments),
            )
            # Imported LRC times are a comparison baseline, not acoustic truth.
            set_task_status(task_id, "aligning", "语音识别完成，正在进行歌词时间轴对齐")
            alignment_started_at = time.perf_counter()
            logger.info("[TASK] 开始歌词匹配 lyrics_lines=%d asr_segments=%d", len(parse_lyrics(clean_lyrics)), len(asr_segments))
            try:
                result = await asyncio.wait_for(
                    asyncio.to_thread(
                        AlignmentPipeline().run,
                        clean_lyrics,
                        asr_segments,
                        imported_lrc or None,
                    ),
                    timeout=ALIGNMENT_TIMEOUT_S,
                )
            except asyncio.TimeoutError as exc:
                logger.exception(
                    "[TASK] 歌词匹配超时 timeout_s=%d elapsed=%.2fs",
                    ALIGNMENT_TIMEOUT_S,
                    time.perf_counter() - alignment_started_at,
                )
                raise RuntimeError(
                    f"歌词匹配超时（{ALIGNMENT_TIMEOUT_S} 秒）"
                ) from exc
            logger.info("[TASK] 歌词匹配完成 elapsed=%.2fs issues=%d", time.perf_counter() - alignment_started_at, result.issue_count)
            set_task_status(task_id, "completed", "识别与对齐已完成")
            logger.info("[TASK] 全部处理完成 mode=%s elapsed=%.2fs", mode, time.perf_counter() - request_started_at)
        except RuntimeError as exc:
            set_task_status(task_id, "failed", "模型加载或识别失败")
            logger.exception("ASR 依赖或模型加载失败")
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except Exception as exc:
            set_task_status(task_id, "failed", "处理失败，请查看错误信息")
            logger.exception("歌词对齐失败")
            raise HTTPException(status_code=500, detail=f"处理失败：{exc}") from exc

    payload = asdict(result)
    payload["issue_count"] = result.issue_count
    payload["audio_filename"] = audio.filename
    payload["transcription_mode"] = mode
    payload["separation_model"] = _separator.model_name if mode == "quality" else None
    payload["asr_segments"] = [asdict(segment) for segment in asr_segments]
    payload["lrc"] = format_lrc(result)
    return payload
