# Unitree G1 异常声音识别系统

确定方案：

```text
模型：EfficientAT DyMN10-AS
训练 GPU：RTX 4090 24GB 云 GPU
预算节奏：50 元训练 v0 -> 接 G1 真机采样 -> 50 元训练 v1
识别类别：呼救声、玻璃破碎声、敲门声、咳嗽声、烟雾报警器声、background
G1 对接：本地实时推理 + audio_event JSON + HTTP webhook 报警
```

当前本机阶段只准备代码、配置、测试、训练脚本和 G1 部署入口。**需要付云 GPU 算力钱时会明确通知，不会自动开始付费训练。**

## 新系统文件

- `config/g1_abnormal_events.yaml`：G1 阈值、连续命中次数、冷却时间、webhook 配置。
- `src/audio_event_poc/event_contract.py`：统一 `audio_event` 输出格式。
- `src/audio_event_poc/runtime.py`：实时分数平滑、阈值判断、冷却去重。
- `src/audio_event_poc/alarm.py`：HTTP webhook 报警适配器。
- `src/audio_event_poc/audioset_manifest.py`：AudioSet 到 5 类异常声的标签映射和采样计划。
- `scripts/g1_fake_realtime_smoke.py`：无需模型的本地事件流 smoke。
- `scripts/g1_realtime_audio_service.py`：G1 实时麦克风推理服务入口。
- `training/efficientat/`：4090 云端训练和 ONNX 导出脚手架。

## 本机先验验证

```powershell
$env:PYTHONPATH="D:\codex\audio_event_poc;D:\codex\audio_event_poc\src"
python scripts/g1_fake_realtime_smoke.py
python -m unittest discover -s tests -v
```

fake smoke 会输出一个标准 `audio_event`，并显示 webhook 未配置时不会发送报警。

## 云 GPU 训练计划

先准备 AudioSet/本地样本元数据：

```powershell
python scripts/prepare_g1_dataset.py build-manifest `
  --metadata-csv data\audioset_g1_metadata.csv `
  --audio-root data\audioset_audio `
  --output-dir data\g1_audio `
  --val-ratio 0.15 `
  --limit distress_call=1200 `
  --limit glass_break=1200 `
  --limit knock=1200 `
  --limit cough=1200 `
  --limit smoke_alarm=1200 `
  --limit background=4000
```

生成 G1 真机采样计划：

```powershell
python scripts/prepare_g1_dataset.py g1-plan `
  --output data\g1_field_collection_plan.csv `
  --per-event 50 `
  --background 150
```

第一轮只花 50 元训练 v0，不追最终精度：

先在云端做 preflight，不启动训练：

```bash
python training/efficientat/cloud_preflight.py \
  --train-manifest data/g1_audio/train_manifest.csv \
  --val-manifest data/g1_audio/val_manifest.csv \
  --efficientat-root /workspace/EfficientAT
```

`ready_for_paid_training` 为 `true`，并确认是 RTX 4090 后，再开始付费训练。这里我会停下来问你。

```bash
python training/efficientat/train_g1_abnormal.py \
  --train-manifest data/g1_audio/train_manifest.csv \
  --val-manifest data/g1_audio/val_manifest.csv \
  --efficientat-root /workspace/EfficientAT \
  --output-dir runs/g1_audio_v0 \
  --epochs 20 \
  --batch-size 48 \
  --freeze-backbone-epochs 3
```

导出：

```bash
python training/efficientat/export_onnx.py \
  --checkpoint runs/g1_audio_v0/best.pt \
  --efficientat-root /workspace/EfficientAT \
  --output models/efficientat_g1_audio_v0.onnx
```

付费训练前先做 dry-run：

```bash
python training/efficientat/train_g1_abnormal.py \
  --train-manifest data/g1_audio/train_manifest.csv \
  --val-manifest data/g1_audio/val_manifest.csv \
  --efficientat-root /workspace/EfficientAT \
  --output-dir runs/g1_audio_v0 \
  --dry-run
```

dry-run 必须确认三件事：manifest 非空、6 类标签合法、EfficientAT 上游目录存在。通过后再开始云 GPU 计费训练。

第二轮在 G1 真机收集误报、漏报和机器人噪声样本后继续训练 v1。

## G1 运行

把 ONNX 模型放到 G1 后运行：

```bash
python scripts/g1_realtime_audio_service.py \
  --model models/efficientat_g1_audio_v0.onnx \
  --config config/g1_abnormal_events.yaml \
  --webhook-url http://127.0.0.1:9000/audio-event
```

如果模型或推理依赖缺失，服务会明确报错退出，不会伪造识别结果。

## 统一 audio_event 示例

```json
{
  "type": "audio_event",
  "schema_version": "1.0",
  "event_key": "smoke_alarm",
  "label": "烟雾报警器声",
  "severity": "critical",
  "confidence": 0.93,
  "threshold": 0.64,
  "source": "unitree_g1_mic",
  "model": "efficientat_dymn10_as_v0",
  "action": {
    "notify_guardian": true,
    "trigger_alarm": true
  }
}
```

## 旧 G1 听觉测试页面

这是一个本地运行的 FastAPI 单页应用，用浏览器录音验证 G1 机器人听觉能力。页面支持人声和环境声识别，包括敲门/敲击、拍手、火警/警报、警笛、咳嗽、流水、键盘敲击、脚步、车辆、音乐、风扇/机器噪声、静音和未知声音。

页面不会 mock 模型输出。OpenAI key、ffmpeg、本地模型文件或推理依赖缺失时，页面和 JSON 都会显示真实错误。

## 安装

建议使用 Python 3.10 或 3.11。先安装 ffmpeg，并确认命令行能执行：

```powershell
ffmpeg -version
```

安装依赖：

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
```

PyTorch/torchaudio 如果需要 CUDA，请按你的 CUDA 版本从 PyTorch 官方源安装对应版本。

## 配置

OpenAI key 从 `.env` 或环境变量读取，不要写进代码：

```env
OPENAI_API_KEY=sk-...
```

本地强模型需要放置这些文件：

```text
panns_data/Cnn14_mAP=0.431.pth
panns_data/class_labels_indices.csv
models/faster-whisper-small-ct2/
```

如果这些文件不存在，`/api/status` 会返回 `local_audio_model_available: false`，本地结果块会显示真实缺失错误。

## 启动

```powershell
python scripts/audio_test_page.py --host 0.0.0.0 --port 8012 --openai-model gpt-audio --realtime-model gpt-realtime-1.5 --local-model models/faster-whisper-small-ct2 --default-backend local
```

打开：

[http://localhost:8012/](http://localhost:8012/)

录音会保存到：

```text
static/audio_test_recordings/
```

事件日志会写入：

```text
logs/audio_test_events.jsonl
```

## 接口

- `GET /`：返回单页 HTML。
- `GET /api/status`：返回 recordings、ffmpeg、OpenAI、本地模型、代理和服务器时间状态。
- `POST /api/recordings`：上传浏览器录音，字段 `file` 和 `backend=local|openai|compare`。
- `GET /api/recordings?limit=20`：返回最近 JSONL 历史记录。
- `POST /api/realtime/calls`：服务端代理 OpenAI Realtime WebRTC SDP 交换，浏览器不会拿到 API key。

## 本地强模型 smoke

生成合成 knock/clap/alarm/cough WAV，并调用 B 本地强模型链路：

```powershell
python scripts/strong_local_event_smoke.py
```

报告输出：

```text
reports/strong_local_event_smoke.json
```

报告中的说明固定写明：合成事件 smoke 只证明链路和标签映射，不等于真实环境精度。

## 旧 PoC 脚本

仓库仍保留原来的命令行音频事件 PoC，例如：

```powershell
python scripts/classify_audio.py --audio samples/knock/knock_001.wav
python scripts/run_once.py --duration 2 --keep-audio samples/latest.wav
```

核心逻辑测试：

```powershell
$env:PYTHONPATH="D:\codex\audio_event_poc;D:\codex\audio_event_poc\src"
python -m unittest discover -s tests -v
```
