#!/bin/bash
#SBATCH -J q-adam-mini
#SBATCH -A L00120230003
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o %j.log
#SBATCH -e %j.err

torchrun --standalone --nproc_per_node 1 my_run_pretrain_halfpre.py \
    --model_config configs/llama_1b.json \
    --continue_from /mntcephfs/data/ruoyusun/hanyizhou/q-adam-mini-checkpoints/model_AdamW-1B_3e-4_256_halfpre_37500 \
    --lr 3e-4 \
    --batch_size 4 \
    --total_batch_size 256 \
    --max_length 1024 \
    --num_billion_training_tokens 13.1 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype float32 \
    --eval_every 2000 \
    --save_every 2500 \
    --single_gpu \
    --optimizer adamw \
    --project '1B' \
    --name AdamW-1B_3e-4_256_halfpre > AdamW_1B_3e-4_256_halfpre_cont2
    
# log文件的作用会被 > 后面的文件取代
# dtype原为bfloat16，现暂时改为float32
# batch_size原为8，现暂时改为4