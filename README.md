# Whisper Convert

Convert MP4 recordings to timestamped text transcripts using [OpenAI Whisper](https://github.com/openai/whisper), then clean up or summarize the result with a local LLM via [LM Studio](https://lmstudio.ai).

## Requirements

- Python 3.10+
- [ffmpeg](https://ffmpeg.org) — `sudo apt install ffmpeg`
- AMD GPU with ROCm (or adjust `device=` in `convert.py` for NVIDIA/CPU)
- [LM Studio](https://lmstudio.ai) running locally (for `format.py`)

## Setup

**1. Install PyTorch with ROCm** (AMD GPU):
```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/rocm7.2
```

For NVIDIA GPU use the standard PyTorch install. For CPU, change `device="cuda"` to `device="cpu"` in `convert.py`.

**2. Install the rest:**
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

**3. Configure `.env`:**
```bash
cp .env.example .env
```

## Configuration

All settings live in `.env`:

| Variable | Default | Description |
|---|---|---|
| `MODEL_SIZE` | `large-v3` | Whisper model: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `WHISPER_LANGUAGE` | `en` | Audio language (ISO 639-1). Leave blank for auto-detect |
| `LLM_BASE_URL` | `http://localhost:1234/v1` | LM Studio server address |
| `LLM_MODEL` | `local-model` | Model ID sent to LM Studio (ignored, uses whatever is loaded) |
| `LLM_TIMEOUT` | `60` | Seconds to wait for LLM response. Increase for large transcripts |
| `LLM_MODE` | `format` | `format` = cleanup only, `summarize` = condense to key points |

Bigger Whisper models are more accurate but slower and use more VRAM:

| Model | VRAM | Speed |
|---|---|---|
| `tiny` | ~1 GB | fastest |
| `base` | ~1 GB | fast |
| `small` | ~2 GB | moderate |
| `medium` | ~5 GB | slow |
| `large-v3` | ~10 GB | slowest, best accuracy |

## Usage

### Step 1 — Transcribe MP4 to text

```bash
python convert.py recording.mp4
# → recording.txt  (timestamped transcript)

python convert.py recording.mp4 -o transcripts/meeting.txt
```

Output format:
```
[00:00:01] Hello, this is the start of the recording.
[00:00:08] And this is the next segment.
```

### Step 2 — Clean up or summarize with LM Studio

Start LM Studio and load a model, then:

```bash
# Remove filler words, fix punctuation, preserve all content
python format.py transcripts/meeting.txt
# → transcripts/meeting_formatted.txt

# Condense to a detailed summary
python format.py transcripts/meeting.txt --mode summarize -o transcripts/meeting_summary.txt
```

For large transcripts (>15K chars) you may need to increase the context window in LM Studio and bump `LLM_TIMEOUT` in `.env`.

## Running tests

```bash
source .venv/bin/activate
pytest tests/ -v
```
