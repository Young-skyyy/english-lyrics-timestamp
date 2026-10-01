from __future__ import annotations

import json
import logging
import wave
from pathlib import Path

from music_tool.lyric_tool import AlignmentPipeline, FasterWhisperTranscriber
from music_tool.lyric_tool.exporters import format_lrc


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)

BASE_DIR = Path(__file__).parent
AUDIO_PATH = BASE_DIR / "test_alignment_audio.wav"
ASR_PATH = BASE_DIR / "test_asr.json"

MODEL_SIZE = "base"
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

lyrics = """夜色落在窗台
我还没有离开
你说故事还很长
我们都别急着悲伤"""


def load_audio_metadata(audio_path: Path) -> dict[str, int]:
    with wave.open(str(audio_path), "rb") as audio:
        frame_count = audio.getnframes()
        sample_rate = audio.getframerate()
        duration_ms = round(frame_count * 1_000 / sample_rate)
        return {
            "duration_ms": duration_ms,
            "sample_rate": sample_rate,
            "channels": audio.getnchannels(),
        }


def main() -> None:
    if not AUDIO_PATH.exists():
        raise FileNotFoundError(
            "测试音频不存在，请先运行："
            " python -m music_tool.examples.generate_test_audio"
        )

    audio_metadata = load_audio_metadata(AUDIO_PATH)
    transcriber = FasterWhisperTranscriber(
        model_size=MODEL_SIZE,
        device=DEVICE,
        compute_type=COMPUTE_TYPE,
    )
    asr_segments = transcriber.transcribe(AUDIO_PATH, language="zh")
    ASR_PATH.write_text(
        json.dumps(
            [segment.__dict__ for segment in asr_segments],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("=== 输入音频 ===")
    print(json.dumps({"path": str(AUDIO_PATH), **audio_metadata}, ensure_ascii=False, indent=2))
    print("\n=== ASR 片段 ===")
    print(json.dumps([segment.__dict__ for segment in asr_segments], ensure_ascii=False, indent=2))

    result = AlignmentPipeline().run(
        lyrics,
        asr_segments,
        """[00:11.98]夜色落在窗台
[00:17.18]我还没有离开
[00:23.58]你说故事还很长
[00:30.00]我们都别急着悲伤""",
    )

    print("\n=== 自动对齐结果 ===")
    print(
        json.dumps(
            {
                "status": result.status,
                "global_offset_ms": result.global_offset_ms,
                "content_confidence": result.content_confidence,
                "timing_confidence": result.timing_confidence,
                "coverage_score": result.coverage_score,
                "issues": [issue.__dict__ for issue in result.issues],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print("\n=== 导出的 LRC ===")
    print(format_lrc(result))


if __name__ == "__main__":
    main()
