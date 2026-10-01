from __future__ import annotations

import logging
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


class DemucsSeparator:
    """使用 Demucs 将完整音频分离为人声轨。"""

    def __init__(
        self,
        model_name: str = "htdemucs",
        device: str = "cpu",
    ) -> None:
        self.model_name = model_name
        self.device = device

    def separate(self, audio_path: str | Path, output_dir: str | Path) -> Path:
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        command = [
            sys.executable,
            "-m",
            "demucs",
            "--name",
            self.model_name,
            "--two-stems=vocals",
            "--device",
            self.device,
            "--out",
            str(output_dir),
            str(audio_path),
        ]
        logger.info(
            "开始人声分离 audio=%s model=%s device=%s",
            audio_path,
            self.model_name,
            self.device,
        )

        try:
            completed = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                "未安装 Demucs，请先运行: python -m pip install demucs"
            ) from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "未知错误").strip()
            raise RuntimeError(f"Demucs 人声分离失败：{detail}") from exc

        if completed.stdout:
            logger.debug("Demucs 输出：%s", completed.stdout.strip())

        vocals_path = output_dir / self.model_name / audio_path.stem / "vocals.wav"
        if not vocals_path.exists():
            raise RuntimeError(f"Demucs 完成但未找到人声文件：{vocals_path}")

        logger.info("人声分离完成 vocals=%s", vocals_path)
        return vocals_path
