# 时间戳精度修复

本次改动基于提交 `a55dc9e`。输出仍然是逐句 LRC；JSON 中保留毫秒时间。

## 修复内容

- 正确读取 faster-whisper 的 `Word.word`，保留模型测得的词起止时间。
- 按整首歌词做动态规划匹配，支持插入词、漏词、重复副歌和同一识别片段中的多行歌词。
- 不再根据字长制造词时间。只有片段时间时，保留片段匹配，并降低时间置信度。
- 跨多个识别片段的歌词保留完整的结束时间；粗粒度匹配不能越过已匹配的词锚点。
- Rap、Quality 模式在 faster-whisper 识别后使用 WhisperX 的 wav2vec2/CTC 做声学时间精校；Quality 先做人声分离。Fast 模式仍只使用 faster-whisper。
- 导入的 LRC 仅用于比较偏差，不再强行把声学时间拉回旧时间轴。前端不再重复应用该偏差。
- 修复 LRC 在 `59.999` 秒处的分钟进位；手动偏移步长从 100ms 调整为 10ms。
- 修复从仓库根目录启动 API 的导入路径，以及 API 缺少 `issue_count` 的问题。

## 运行

在仓库根目录执行（Python 3.10+）：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn api:app --reload
```

打开 <http://127.0.0.1:8000>。Rap 和 Quality 首次运行会加载/下载 CTC 模型，需要联网，耗时会增加。Rap 识别模型仍默认使用 `base`，可通过 `LYRICFLOW_RAP_MODEL` 指向其他 faster-whisper 模型或本地路径。程序尊重用户自行设置的离线环境变量。

## 验证

核心回归测试不需要下载语音模型：

```powershell
python -m unittest discover -s tests -v
node tests/test_web_precision.cjs
```

API 集成测试另外需要 `httpx`：`pip install httpx`。未安装 FastAPI/httpx 时该测试自动跳过。测试中的模型输出使用固定数据和模拟接口，覆盖词字段读取、跨片段边界、重复副歌、长前奏、缩写、声学精校接线、空结果重试、旧 LRC 偏差和网页二次偏移。

这些测试验证代码行为，不代表真实歌曲的毫秒误差评测。当前验证环境没有安装 PyTorch、faster-whisper 或 WhisperX，也没有运行真实模型。实际精度仍需用歌曲音频、对应歌词和人工标注时间验证，不能承诺固定误差上限。LRC 的两位小数是存储分辨率，不是模型准确率。
