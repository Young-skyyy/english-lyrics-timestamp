from __future__ import annotations

import argparse
import logging

from .exporters import format_json, format_lrc
from .models import AsrSegment
from .pipeline import AlignmentPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="歌词自动对齐与异常检测工具")
    parser.add_argument("--lyrics", required=True, help="纯文本歌词文件")
    parser.add_argument("--output", default="result.json", help="JSON 输出路径")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s - %(message)s")
    with open(args.lyrics, encoding="utf-8") as file:
        lyrics = file.read()
    raise SystemExit("CLI 当前需要接入真实 ASR；请先使用 examples/demo.py 验证流程。")
