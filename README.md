# English Lyrics Timestamp

英文音乐识别与歌词时间戳工具。项目用于将歌曲音频转录为歌词，并把已有歌词与实际演唱时间进行对齐，生成可用于播放器显示的同步歌词。

## 主要功能

- 英文歌曲语音识别；
- 歌词与音频时间轴对齐；
- 词级或句级时间戳处理；
- LRC、JSON 等结果导出；
- 识别置信度与对齐结果分析；
- 可选的人声分离和高精度识别模式；
- FastAPI Web 接口和浏览器页面；
- 示例音频、歌词和 ASR 数据。

## 项目结构

```text
lyric_tool/       核心识别、匹配、对齐和导出模块
web/              Web 页面
examples/         示例音频、歌词和测试数据
api.py            FastAPI 服务
requirements.txt  Python 依赖
```

## 环境要求

- Python 3.10 或更高版本；
- 建议使用虚拟环境；
- 语音识别和人声分离功能首次运行时可能需要下载模型；
- `whisperx`、`demucs` 等依赖对 PyTorch 和系统音频环境有要求。

## 安装

在仓库父目录执行：

```powershell
cd english-lyrics-timestamp
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

如果 PowerShell 阻止脚本执行，可以先运行：

```powershell
Set-ExecutionPolicy -Scope Process Bypass
```

## 启动 Web 服务

在仓库父目录执行：

```powershell
uvicorn api:app --reload
```

然后打开：

<http://127.0.0.1:8000>

健康检查接口：

```text
GET http://127.0.0.1:8000/api/health
```

Web 页面可以上传音频和歌词，并选择不同的识别模式：

- `fast`：普通快速识别；
- `rap`：针对快速演唱的识别与时间校正；
- `quality`：先进行人声分离，再进行识别，处理时间更长。

## 示例数据

示例文件位于 `examples/`：

- `english_test_audio.wav`：英文测试音频；
- `english_test_lyrics.txt`：纯文本歌词；
- `english_test.lrc`：带时间戳的歌词；
- `test_asr.json`：测试用 ASR 结果。

## Python 使用示例

```python
from lyric_tool import AlignmentPipeline

# 具体的 ASR 结果需要先转换为项目中的 AsrSegment 对象。
# AlignmentPipeline 可用于把识别片段与歌词进行匹配和对齐。
result = AlignmentPipeline().run(lyrics, asr_segments, imported_lrc=None)
```

## 当前状态

Web API 已支持完整的上传、识别、歌词匹配和 LRC 输出流程。命令行入口目前主要用于保留 CLI 接口，真实 ASR 流程建议通过 Web API 或示例程序验证。

## 注意事项

- 请勿将个人 API Key、模型缓存或本地环境文件提交到仓库；
- 音频文件可能包含受版权保护的内容，请确保拥有合法使用权限；
- 大型模型首次加载可能需要较长时间和较多内存；
- 项目中的示例音频仅用于本地测试。

## License

当前仓库未单独声明许可证。公开发布或再分发前，请根据需要补充许可证文件。

## Timestamp precision update

See [PRECISION.md](PRECISION.md) for fixes, model requirements and regression tests.
