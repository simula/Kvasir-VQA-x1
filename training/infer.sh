#!/usr/bin/env bash
# Run inference with a fine-tuned adapter (or a base model) on a test JSONL.
# Edit MODEL / ADAPTER / VAL / RESULT for your run. Adapters can be a local
# checkpoint dir or a Hub id, e.g. SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft.
set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
ADAPTER="${ADAPTER:-SimulaMet/Qwen2.5-VL-KvasirVQA-x1-ft}"   # empty string => base model only
VAL="${VAL:-data/1_transform_to_vqa_format_test.jsonl}"
RESULT="${RESULT:-results/pred_qwen25vl_ft.jsonl}"

ADAPTER_ARGS=()
if [ -n "${ADAPTER}" ]; then ADAPTER_ARGS=(--adapters "${ADAPTER}"); fi

mkdir -p "$(dirname "${RESULT}")"

MAX_PIXELS=432000 \
swift infer \
    --model "${MODEL}" \
    "${ADAPTER_ARGS[@]}" \
    --use_hf True \
    --system 'You are a medical vision-language assistant; given an endoscopic image and a clinical question that may ask about one or more findings, provide a concise, clinically accurate response addressing all parts of the question in natural-sounding medical language as if spoken by a doctor in a single sentence.' \
    --infer_backend lmdeploy \
    --max_model_len 1100 \
    --max_new_tokens 512 \
    --logprobs True \
    --val_dataset "${VAL}" \
    --result_path "${RESULT}"
