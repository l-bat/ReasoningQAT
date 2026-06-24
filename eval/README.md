# Evaluation

This directory contains the evaluation infrastructure for reproducing the paper's benchmark results using [evalscope](https://github.com/modelscope/evalscope) and a [vLLM](https://github.com/vllm-project/vllm) server.

## Structure

```
eval/
├── evalscope_vllm.py          # CLI wrapper: runs evalscope against a vLLM server
├── yamls/vllm/
│   ├── Qwen3/
│   │   ├── gsm8k.yaml
│   │   ├── math500.yaml
│   │   ├── aime.yaml
│   │   ├── gpqa.yaml
│   │   └── livecode.yaml
└── results/                   # evaluation outputs (created at runtime)
```

## Prerequisites

```bash
# Activate the project environment
source env/bin/activate

# Required packages
pip install evalscope vllm
```

> **CUDA 13.0+ note:** vLLM's FlashInfer sampler fails to compile on CUDA 13.0 due to a CUB API change. Always set `VLLM_USE_FLASHINFER_SAMPLER=0` when starting the server.

## Paper evaluation settings

All benchmarks use the following generation parameters (Section 4 of the paper):

| Parameter | Value |
|-----------|-------|
| Temperature | 0.6 |
| Top-p | 0.95 |
| Max tokens | 32,768 |
| Seeds | 3 (averaged) |

---

## Step 1: Start the vLLM server

The evaluation requires a running vLLM OpenAI-compatible server. Start it before running any eval script.

### HuggingFace model

```bash
CUDA_VISIBLE_DEVICES=0,1,2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --trust-remote-code \
  --port 8000 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --data-parallel-size 3   # use as many free GPUs as available
```

### Local checkpoint (e.g. fine-tuned or quantized model)

```bash
MODEL_PATH="/path/to/your/local/model"
MODEL_NAME="my-model"   # name used in --served-model-name and --model below

CUDA_VISIBLE_DEVICES=0,1,2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --port 8000 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --data-parallel-size 3
```

> `--dtype bfloat16` is recommended for fp32 checkpoints to halve GPU memory usage.

### Stripped NNCF checkpoint

NNCF QAT checkpoints saved via [save_stripped.py](https://github.com/l-bat/nncf/blob/lt/3bit_equalization/examples/llm_compression/torch/distillation_qat_with_lora/save_stripped.py) are standard HF-format safetensors files.

**Step 1 — start the server**:

```bash
export MODEL_PATH="/path/to/stripped_nncf_checkpoint"
export MODEL_NAME="stripped_nncf_checkpoint"   # short name, alphanumeric/-/_/. only

CUDA_VISIBLE_DEVICES=0,1,2 VLLM_USE_FLASHINFER_SAMPLER=0 \
  vllm serve "$MODEL_PATH" \
  --served-model-name "$MODEL_NAME" \
  --trust-remote-code \
  --port 8000 \
  --max-model-len 40960 \
  --gpu-memory-utilization 0.9 \
  --dtype bfloat16 \
  --data-parallel-size 3
```

**Step 2 — run evaluation** (use `$MODEL_NAME` for both `--model` and `--model-id`):

```bash
python eval/evalscope_vllm.py eval/yamls/vllm/Qwen3/gsm8k.yaml \
  --model "$MODEL_NAME" \
  --model-id "$MODEL_NAME" \
  --api-url "http://127.0.0.1:8000/v1/chat/completions" \
  --work-dir "eval/results/${MODEL_NAME}/gsm8k_32k" \
  --generation-config '{"seed": 42, "max_tokens": 32768}'
```

Results will be saved under `eval/results/<MODEL_NAME>/`.

### Verify the server is ready

```bash
curl -s http://127.0.0.1:8000/v1/models
# Expected: {"data": [{"id": "<model-name>", ...}]}
```

---

## Step 2: Run a benchmark

```bash
cd /path/to/ReasoningQAT
source env/bin/activate

python eval/evalscope_vllm.py eval/yamls/vllm/Qwen3/<benchmark>.yaml \
  --model <served-model-name> \
  --model-id <model-id> \
  --api-url "http://127.0.0.1:8000/v1/chat/completions" \
  --work-dir "eval/results/<model-id>/<benchmark>_32k" \
  --generation-config '{"seed": 42, "max_tokens": 32768}'
```

### Available benchmarks

| `<benchmark>` | Dataset | # Problems |
|---------------|---------|-----------|
| `gsm8k` | GSM8K | 1,319 |
| `math500` | MATH-500 | 500 |
| `aime` | AIME 2024–2025 | 60 (see note below) |
| `gpqa` | GPQA-Diamond | 198 |
| `livecode` | LiveCodeBench (`release_latest`) | varies |

> **AIME note:** The paper reports AIME-120 (2022–2025). Currently only `aime24` and `aime25` (60 problems) are available in evalscope. AIME 2022–2023 splits will be added once available upstream.

### 3-seed evaluation (to match paper)

Run three times with different seeds and average the scores:

```bash
for SEED in 42 123 456; do
  python eval/evalscope_vllm.py eval/yamls/vllm/Qwen3/gsm8k.yaml \
    --model Qwen3-4B \
    --model-id Qwen3-4B \
    --api-url "http://127.0.0.1:8000/v1/chat/completions" \
    --work-dir "eval/results/Qwen3-4B/gsm8k_32k_seed${SEED}" \
    --generation-config "{\"seed\": ${SEED}, \"max_tokens\": 32768}"
done
```

### Full benchmark suite for one model

```bash
MODEL="Qwen3-4B"
API_URL="http://127.0.0.1:8000/v1/chat/completions"

for BENCHMARK in gsm8k math500 aime gpqa livecode; do
  for SEED in 42 123 456; do
    python eval/evalscope_vllm.py eval/yamls/vllm/Qwen3/${BENCHMARK}.yaml \
      --model "$MODEL" \
      --model-id "$MODEL" \
      --api-url "$API_URL" \
      --work-dir "eval/results/${MODEL}/${BENCHMARK}_32k_seed${SEED}" \
      --generation-config "{\"seed\": ${SEED}, \"max_tokens\": 32768}"
  done
done
```

## Results

Evaluation outputs are saved under `eval/results/<model-id>/<benchmark>/` in JSON format, including per-sample predictions and aggregate accuracy scores.
