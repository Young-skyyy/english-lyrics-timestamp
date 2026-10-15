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

这些回归测试验证代码行为，不代表真实歌曲的毫秒误差评测。LRC 的两位小数是存储分辨率，不是模型准确率。

## 真实音频验证（2026-10-15）

使用用户提供的《Not Like Us》音频（274.204 秒）和 115 行歌词，通过实际 `/api/align` 请求完成 Fast 与 Rap 两种模型流程。运行环境为 Python 3.10.11、faster-whisper 1.2.1、WhisperX 3.8.6、PyTorch/torchaudio 2.8.0，CPU int8，Whisper 模型为 `base`。Rap 使用已缓存的英文 wav2vec2 CTC 模型。

实测发现并修复了缺失歌词向音频尾部连续外推、前端为缺失歌词制造时间、以及估算时间虚增匹配覆盖率的问题。现在无声学依据的行保留空时间，网页显示 `--:--.--`，LRC 导出跳过这些行；JSON 保留完整歌词和待审核状态。声学词置信度也会限制时间置信度。片段候选的最低匹配得分调整为 0.55。

保留真实模型输出、用修复后的对齐器重算后，Fast 输出 87/115 行时间（75.65%），Rap 输出 106/115 行时间（92.17%）；两份结果均无时间倒退或超出音频范围的行。Rap 另有 9 行未定时，生成时间的行仍需审核。两种模式的比较同时涉及 VAD、beam 和 CTC 设置。

这次没有人工时间标注，因此上述数字是匹配覆盖情况，无法据此计算毫秒误差或声称准确率。音频、歌词及逐句结果仅保存在用户本地，没有提交到仓库。Quality 人声分离模式没有进行真实音频实测。

已通过 21 项测试（包含 API 集成测试，无跳过）及网页时间轴检查。

可用自带脚本重复真实请求（需安装项目依赖与 `httpx`）：

```powershell
python examples/validate_audio.py --audio song.mp3 --lyrics clean-lyrics.txt --mode rap --duration-ms 274204 --output-dir validation-output
```

脚本输出 LRC、完整 JSON 和诊断摘要；未知音频时长可省略 `--duration-ms`。歌词文本应去掉歌曲标题，段落标签可保留。模型缓存位置可通过标准 `HF_HOME`、`HF_HUB_CACHE`、`TORCH_HOME` 环境变量配置。
