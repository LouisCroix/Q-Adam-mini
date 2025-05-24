#!/bin/bash
#SBATCH -J q-adam-mini
#SBATCH -A L00120230003
#SBATCH -p p-A800
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:4
#SBATCH -o %j.log
#SBATCH -e %j.err

# python my_run_finetune.py \
#     --model ./.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
#     --lr 2e-6 \
#     --batch_size 2 \
#     --total_batch_size 80 \
#     --max_length 2048 \
#     --num_epochs 3 \
#     --weight_decay 0 \
#     --dtype float32 \
#     --eval_every 50 \
#     --optimizer adammini \
#     --project 'finetune' \
#     --name Adam_mini_new1_llama_2e-6_80_halfpre >  Adam_mini_new1_llama_2e-6_80_halfpre

accelerate launch ../my_run_finetune.py \
    --model ./.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
    --task gsm \
    --lr 2e-6 \
    --batch_size 2 \
    --total_batch_size 80 \
    --max_length 2048 \
    --num_epochs 3 \
    --weight_decay 0 \
    --dtype float32 \
    --eval_every 50 \
    --optimizer adammini \
    --project 'finetune_gsm' \
    --name Adam_mini_new1_llama_gsm_2e-6_80_halfpre >  Adam_mini_new1_llama_gsm_2e-6_80_halfpre

# log文件的作用会被 > 后面的文件取代
# dtype原为bfloat16，现暂时改为float32