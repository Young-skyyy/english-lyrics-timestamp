from __future__ import annotations

import multiprocessing as mp
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any


class _WorkerFailure(RuntimeError):
    pass


def _worker_main(mode: str, connection: Any) -> None:
    try:
        connection.send(("stage", "worker_started"))
        if mode == "rap":
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
            connection.send(("stage", "import_faster_whisper_transcriber"))
            from .transcriber import FasterWhisperTranscriber

            connection.send(("stage", "create_rap_transcriber"))
            transcriber = FasterWhisperTranscriber(
                model_size=os.getenv("LYRICFLOW_RAP_MODEL", "base"),
                device="cpu",
                compute_type="int8",
            )
        else:
            connection.send(("stage", "import_faster_whisper_transcriber"))
            from .transcriber import FasterWhisperTranscriber

            transcriber = FasterWhisperTranscriber(
                model_size="base",
                device="cpu",
                compute_type="int8",
            )
        connection.send(("ready", None))
    except BaseException:
        connection.send(("error", traceback.format_exc()))
        connection.close()
        return

    try:
        while True:
            command, args = connection.recv()
            if command == "close":
                return
            try:
                if mode == "rap":
                    result = transcriber.transcribe(
                        args[0],
                        language="en",
                        beam_size=8,
                        vad_filter=False,
                    )
                else:
                    result = transcriber.transcribe(*args)
                connection.send(("result", result))
            except BaseException:
                connection.send(("error", traceback.format_exc()))
    finally:
        connection.close()


class IsolatedTranscriber:
    """在独立进程中持有 ASR 模型，使超时可以真正终止底层推理。"""

    def __init__(self, mode: str) -> None:
        if mode not in {"fast", "quality", "rap"}:
            raise ValueError(f"不支持的转录模式: {mode}")
        self.mode = mode
        self._context = mp.get_context("spawn")
        self._lock = threading.Lock()
        self._process: mp.Process | None = None
        self._connection: Any = None

    def _stop_locked(self) -> None:
        process = self._process
        connection = self._connection
        self._process = None
        self._connection = None
        if connection is not None:
            connection.close()
        if process is None:
            return
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive() and hasattr(process, "kill"):
            process.kill()
            process.join(2)
        if not process.is_alive() and process.exitcode is not None:
            process.close()

    def _start_locked(self, timeout: float) -> None:
        parent, child = self._context.Pipe(duplex=True)
        process = self._context.Process(target=_worker_main, args=(self.mode, child))
        process.daemon = True
        process.start()
        child.close()
        self._process = process
        self._connection = parent
        deadline = time.perf_counter() + timeout
        last_stage = "process_started"
        while True:
            remaining = max(0.0, deadline - time.perf_counter())
            if not parent.poll(min(1.0, remaining)):
                if remaining <= 0:
                    self._stop_locked()
                    raise TimeoutError(
                        f"模型加载超时（{int(timeout)} 秒），最后阶段={last_stage}"
                    )
                continue
            status, detail = parent.recv()
            if status == "stage":
                last_stage = str(detail)
                continue
            if status != "ready":
                self._stop_locked()
                raise _WorkerFailure(detail or "转录器子进程启动失败")
            return

    def transcribe(
        self,
        audio_path: str | Path,
        language: str | None,
        lyrics: str,
        model_load_timeout: float,
        inference_timeout: float,
    ) -> list[Any]:
        args = (
            (audio_path, lyrics)
            if self.mode == "rap"
            else (audio_path, language)
        )
        with self._lock:
            started_at = time.perf_counter()
            if self._process is None or not self._process.is_alive():
                self._stop_locked()
                self._start_locked(model_load_timeout)
            try:
                self._connection.send(("transcribe", args))
                if not self._connection.poll(inference_timeout):
                    elapsed = time.perf_counter() - started_at
                    self._stop_locked()
                    raise TimeoutError(
                        f"语音识别超时（{int(inference_timeout)} 秒），已终止推理子进程，elapsed={elapsed:.2f}s"
                    )
                status, detail = self._connection.recv()
                if status == "error":
                    raise _WorkerFailure(detail or "转录器子进程执行失败")
                if status != "result":
                    raise _WorkerFailure("转录器子进程返回了未知消息")
                return detail
            except (EOFError, BrokenPipeError, OSError) as exc:
                self._stop_locked()
                raise _WorkerFailure("转录器子进程已意外退出") from exc

    def close(self) -> None:
        with self._lock:
            if self._connection is not None and self._process is not None and self._process.is_alive():
                try:
                    self._connection.send(("close", None))
                except (BrokenPipeError, OSError):
                    pass
            self._stop_locked()


_runners: dict[str, IsolatedTranscriber] = {}
_runners_lock = threading.Lock()


def run_transcription(
    mode: str,
    audio_path: str | Path,
    language: str | None,
    lyrics: str,
    model_load_timeout: float,
    inference_timeout: float,
) -> list[Any]:
    with _runners_lock:
        runner = _runners.setdefault(mode, IsolatedTranscriber(mode))
    return runner.transcribe(
        audio_path,
        language,
        lyrics,
        model_load_timeout,
        inference_timeout,
    )
