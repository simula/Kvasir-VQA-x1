#!/usr/bin/env bash
# Fine-tune MedGemma-4B-it with LoRA via MS-Swift (Track 2: transformed images).
# Targets language-model projections only and retrains embed/lm_head.
set -euo pipefail

WANDB_PROJECT="Kvasir-VQA-x1" \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
NPROC_PER_NODE=4 \
MAX_PIXELS=432000 \
swift sft \
    --model google/medgemma-4b-it \
    --dataset 'data/2_transform_to_vqa_format_train.jsonl' \
    --system 'You are a medical vision-language assistant; given an endoscopic image and a clinical question that may ask about one or more findings, provide a concise, clinically accurate response addressing all parts of the question in natural-sounding medical language as if spoken by a doctor in a single sentence.' \
    --use_hf True \
    --train_type lora \
    --torch_dtype bfloat16 \
    --num_train_epochs 4 \
    --per_device_train_batch_size 12 \
    --per_device_eval_batch_size 8 \
    --learning_rate 2e-5 \
    --lr_scheduler_type linear \
    --lora_rank 16 \
    --lora_alpha 64 \
    --freeze_vit true \
    --gradient_accumulation_steps 3 \
    --eval_steps 500 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 1000 \
    --output_dir output/medgemma \
    --warmup_ratio 0.03 \
    --dataloader_num_workers 8 \
    --dataset_num_proc 8 \
    --report_to wandb \
    --deepspeed zero2 \
    --target_regex "model\.language_model\.layers\.\d+\..*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)" \
    --modules_to_save embed_tokens,lm_head
