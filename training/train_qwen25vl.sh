#!/usr/bin/env bash
# Fine-tune Qwen2.5-VL-7B-Instruct with LoRA via MS-Swift (Track 1: original images).
# Requires: conda env with ms-swift, deepspeed; 4-8 GPUs.
# For the transformed track, point --dataset at 2_transform_to_vqa_format_train.jsonl.
set -euo pipefail

WANDB_PROJECT="Kvasir-VQA-x1" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_PIXELS=432000 \
swift sft \
    --model Qwen/Qwen2.5-VL-7B-Instruct \
    --dataset 'data/1_transform_to_vqa_format_train.jsonl' \
    --system 'You are a medical vision-language assistant; given an endoscopic image and a clinical question that may ask about one or more findings, provide a concise, clinically accurate response addressing all parts of the question in natural-sounding medical language as if spoken by a doctor in a single sentence.' \
    --use_hf True \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 32 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --lora_rank 16 \
    --lora_alpha 64 \
    --target_modules all-linear \
    --freeze_vit true \
    --gradient_accumulation_steps 1 \
    --eval_steps 500 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 1000 \
    --output_dir output/qwen25vl \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --report_to wandb \
    --deepspeed zero2
