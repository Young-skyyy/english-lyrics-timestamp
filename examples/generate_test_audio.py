from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path


SAMPLE_RATE = 16_000
AUDIO_PATH = Path(__file__).with_name("test_alignment_audio.wav")
ASR_PATH = Path(__file__).with_name("test_asr.json")
LYRICS_PATH = Path(__file__).with_name("english_test_lyrics.txt")

# 这些音调片段仅用于保留原有时间轴回归测试。
SEGMENTS = [
    (12_400, 16_850, 220.0),
    (17_600, 21_400, 246.94),
    (24_560, 28_500, 261.63),
    (30_420, 34_600, 293.66),
]
TEXTS = [
    "The night is falling on the window",
    "I am still here waiting for you",
    "You said the story is not over",
    "We can leave the past behind",
]


def envelope(position: int, length: int) -> float:
    attack = min(1.0, position / max(1, int(SAMPLE_RATE * 0.08)))
    release = min(1.0, (length - position) / max(1, int(SAMPLE_RATE * 0.12)))
    return max(0.0, min(attack, release))


def generate_audio() -> None:
    duration_ms = 36_000
    total_samples = int(duration_ms * SAMPLE_RATE / 1_000)
    samples = [0.0] * total_samples

    for start_ms, end_ms, frequency in SEGMENTS:
        start = int(start_ms * SAMPLE_RATE / 1_000)
        end = int(end_ms * SAMPLE_RATE / 1_000)
        length = end - start
        for position, sample_index in enumerate(range(start, end)):
            time_s = position / SAMPLE_RATE
            tone = (
                0.34 * math.sin(2 * math.pi * frequency * time_s)
                + 0.16 * math.sin(2 * math.pi * frequency * 2 * time_s)
                + 0.08 * math.sin(2 * math.pi * frequency * 3 * time_s)
            )
            samples[sample_index] = tone * envelope(position, length)

    pcm = b"".join(
        struct.pack("<h", int(max(-1.0, min(1.0, sample)) * 32_767))
        for sample in samples
    )
    with wave.open(str(AUDIO_PATH), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(SAMPLE_RATE)
        audio.writeframes(pcm)

    asr_data = [
        {
            "text": text,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "confidence": confidence,
        }
        for text, (start_ms, end_ms, _), confidence in zip(
            TEXTS, SEGMENTS, [0.96, 0.93, 0.91, 0.95]
        )
    ]
    ASR_PATH.write_text(json.dumps(asr_data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"generated: {AUDIO_PATH}")
    print(f"generated: {ASR_PATH}")


if __name__ == "__main__":
    generate_audio()
