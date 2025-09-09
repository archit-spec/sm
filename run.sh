
#!/bin/bash
export HF_HOME=/workspace/.cache/huggingface
# Set memory optimization environment variables
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8
source .venv/bin/activate
# Conservative settings for initial run
uv run pretrain.py \
    --model_name Qwen/Qwen3-14B-Base \
    --dataset_name archit11/hyperswitch-code-only \
    --content_field content \
    --output_dir ./pretrained_qwen_14b \
    --multi_gpu \
    --batch_size=4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 1 \
    --learning_rate 1e-6 \
    --max_length 3092 \
    --warmup_steps 15 \
    --save_steps 200 \
    --logging_steps 1 \
    --bf16 \
    --use_wandb