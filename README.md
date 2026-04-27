# Meeting ASR Pipeline

Local-first meeting transcription pipeline with Qwen3-ASR, lightweight speaker diarization, and a continuity-preserving long-window ASR fallback.

This repository is built for long Chinese meeting recordings where both speaker attribution and transcript completeness matter. It runs fully on your own GPU server, keeps model environments isolated, and avoids uploading private audio to external APIs.

## Highlights

- **Local ASR**: runs Qwen3-ASR from local model weights.
- **Speaker attribution**: uses VAD, SpeechBrain ECAPA speaker embeddings, and clustering.
- **Long-recording friendly**: supports hour-long recordings through chunked ASR.
- **Two-route design**: produces both speaker-attributed transcripts and long-window continuity transcripts.
- **Mainland China friendly**: documents ModelScope, Hugging Face mirror, and PyPI mirror usage.
- **Private by default**: raw audio, generated outputs, caches, and model weights are excluded from Git.

## Architecture

```text
Route A: speaker-attributed transcript

meeting.wav
  -> normalize to 16 kHz mono
  -> energy VAD
  -> speaker embedding windows
  -> KMeans speaker clustering
  -> speaker-aware audio chunks
  -> Qwen3-ASR
  -> final_transcript.md / final_transcript.jsonl

Route B: continuity transcript

meeting.wav
  -> normalize to 16 kHz mono
  -> fixed overlapping windows
  -> Qwen3-ASR
  -> full_window_asr/final_transcript.md / final_transcript.jsonl
```

Use Route A as the main readable transcript. Use Route B to audit content near chunk boundaries and recover context that may be weakened by diarization splits.

## Repository Layout

```text
.
├── README.md
├── .env.example
├── .gitignore
├── data/
│   └── raw_input/
│       └── .gitkeep
├── docs/
│   └── OPEN_SOURCE_CHECKLIST.md
├── envs/
│   ├── qwen3-asr.yml
│   └── speaker-diarization.yml
├── outputs/
│   └── .gitkeep
└── scripts/
    ├── check_env.py
    ├── diarize_meeting.py
    ├── export_fixed_windows.py
    ├── transcribe_manifest_qwen3_asr.py
    └── transcribe_qwen3_asr.py
```

## Installation

Create separate conda environments for ASR and speaker diarization. This avoids dependency conflicts between `qwen-asr`, `transformers`, `speechbrain`, `torch`, and `huggingface_hub`.

```bash
conda env create -f envs/qwen3-asr.yml
conda env create -f envs/speaker-diarization.yml
```

If you are on a mainland China server, configure mirrors before installing:

```bash
export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
```

## Model Weights

Model weights are not included in this repository. Download them into a local model directory, for example:

```bash
mkdir -p models

modelscope download \
  --model Qwen/Qwen3-ASR-1.7B \
  --local_dir models/Qwen3-ASR-1.7B

HF_ENDPOINT=https://hf-mirror.com huggingface-cli download \
  speechbrain/spkrec-ecapa-voxceleb \
  --local-dir models/spkrec-ecapa-voxceleb
```

The scripts read these defaults:

```bash
MEETING_ASR_QWEN_MODEL_DIR=models/Qwen3-ASR-1.7B
MEETING_ASR_SPEAKER_MODEL_DIR=models/spkrec-ecapa-voxceleb
```

You can override them with environment variables or with CLI arguments such as `--model-dir` and `--embedding-model`.

## GPU Policy

Check GPU availability before running inference:

```bash
nvidia-smi
```

Choose an idle physical GPU with `CUDA_VISIBLE_DEVICES`. Inside the Python script, the selected physical GPU is visible as `cuda:0`.

```bash
export GPU_ID=0
```

## Quick Start

Place a private recording under `data/raw_input/`. This directory is ignored by Git.

```bash
export INPUT_AUDIO="data/raw_input/meeting.wav"
export RUN_DIR="outputs/meeting_run"
export GPU_ID=0
```

### 1. Speaker-attributed Route

Run diarization and export speaker-aware ASR chunks:

```bash
conda activate speaker-diarization

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/diarize_meeting.py \
  "${INPUT_AUDIO}" \
  --output-dir "${RUN_DIR}" \
  --num-speakers 2 \
  --device cuda:0 \
  --batch-size 96 \
  --max-chunk-sec 28 \
  --cluster-method kmeans
```

Run Qwen3-ASR on the speaker chunks:

```bash
conda activate qwen3-asr

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/transcribe_manifest_qwen3_asr.py \
  --manifest "${RUN_DIR}/asr_manifest.json" \
  --output-dir "${RUN_DIR}" \
  --language Chinese \
  --batch-size 16 \
  --max-new-tokens 512
```

Outputs:

```text
outputs/meeting_run/final_transcript.md
outputs/meeting_run/final_transcript.jsonl
```

### 2. Continuity Route

Export fixed overlapping windows:

```bash
conda activate speaker-diarization

python scripts/export_fixed_windows.py \
  "${RUN_DIR}/audio_16k_mono.wav" \
  --output-dir "${RUN_DIR}" \
  --window-sec 300 \
  --overlap-sec 30 \
  --manifest-name full_window_manifest.json
```

Run Qwen3-ASR on the long windows:

```bash
conda activate qwen3-asr

CUDA_VISIBLE_DEVICES=${GPU_ID} python scripts/transcribe_manifest_qwen3_asr.py \
  --manifest "${RUN_DIR}/full_window_manifest.json" \
  --output-dir "${RUN_DIR}/full_window_asr" \
  --language Chinese \
  --batch-size 1 \
  --max-new-tokens 4096
```

Outputs:

```text
outputs/meeting_run/full_window_asr/final_transcript.md
outputs/meeting_run/full_window_asr/final_transcript.jsonl
```

## Output Files

| File | Purpose |
| --- | --- |
| `audio_16k_mono.wav` | Normalized mono audio used by downstream steps |
| `speech_segments.json` | Energy-VAD speech spans |
| `embedding_windows.json` | Speaker embedding windows and assigned speaker IDs |
| `diarization_segments.json` | Speaker-attributed time spans |
| `asr_manifest.json` | Per-speaker ASR chunk manifest |
| `final_transcript.md` | Speaker-route Markdown transcript |
| `final_transcript.jsonl` | Speaker-route machine-readable transcript |
| `full_window_manifest.json` | Fixed-window ASR manifest |
| `full_window_asr/final_transcript.md` | Continuity-route Markdown transcript |
| `full_window_asr/final_transcript.jsonl` | Continuity-route machine-readable transcript |

## Recommended Defaults

For a one-hour, two-speaker Chinese meeting on an A100-class GPU:

```text
diarization:
  num_speakers: 2
  cluster_method: kmeans
  max_chunk_sec: 28
  batch_size: 96

speaker ASR:
  batch_size: 16
  max_new_tokens: 512

continuity ASR:
  window_sec: 300
  overlap_sec: 30
  batch_size: 1
  max_new_tokens: 4096
```

If continuity ASR is too slow, reduce `--window-sec` to `180`. If boundary loss is still visible, increase `--overlap-sec` to `45` or `60`.

## Notes

- SpeechBrain `1.0.3` should use `huggingface_hub==0.36.2`.
- `diarize_meeting.py` loads SpeechBrain with `overrides={"pretrained_path": model_dir}` to avoid unexpected remote fetches.
- Whole-audio ASR can be impractical for hour-long recordings. The fixed-window continuity route is the recommended fallback.
- Stereo channels are not assumed to be separate speakers. The pipeline mixes input to mono before diarization.

## Open Source Safety

Before publishing to GitHub, read [docs/OPEN_SOURCE_CHECKLIST.md](docs/OPEN_SOURCE_CHECKLIST.md).

This repository intentionally ignores:

- raw audio under `data/raw_input/`
- generated transcripts and chunks under `outputs/`
- local model weights under `models/` or `PretrainedModels/`
- local environments and caches such as `.venv/` and `.uv-cache/`
- private `.env` files

Do not use `git add .` until `git status --ignored` confirms private artifacts are excluded.

## Repository Name

Recommended GitHub repository slug:

```text
meeting-asr-pipeline
```

Use `Meeting ASR Pipeline` as the display title in README and project descriptions.

## License

MIT. See [LICENSE](LICENSE).
