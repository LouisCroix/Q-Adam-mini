#!/bin/bash
#SBATCH -J q-adam-mini
#SBATCH -A L00120230003
#SBATCH -p p-A100
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1
#SBATCH -o %j.log
#SBATCH -e %j.err

torchrun --standalone --nproc_per_node 1 my_run_finetune.py \
    --model /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
    --lr 2e-6 \
    --batch_size 1 \
    --total_batch_size 80 \
    --max_length 2048 \
    --num_epochs 3 \
    --weight_decay 0 \
    --dtype float32 \
    --single_gpu \
    --eval_every 50 \
    --optimizer q_adam_mini_8bit \
    --project 'finetune' \
    --weight_group_size 256 \
    --weight_quant \
    --stochastic_round \
    --stochastic_round_state \
    --name new1_llama_2e-6_80_halfpre > new1_llama_2e-6_80_halfpre

# deepspeed --num_gpus=1 my_run_finetune.py \
#     --deepspeed \
#     --deepspeed_config /home/hanyizhou/Q-Adam-mini/ds_configs/config.json \
#     --model /mntcephfs/data/ruoyusun/liziniu/.cache/huggingface/hub/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
#     --lr 2e-6 \
#     --batch_size 1 \
#     --total_batch_size 80 \
#     --max_length 2048 \
#     --num_epochs 3 \
#     --weight_decay 0 \
#     --dtype float32 \
#     --single_gpu \
#     --eval_every 50 \
#     --optimizer q_adam_mini_8bit \
#     --project 'finetune' \
#     --weight_group_size 256 \
#     --weight_quant \
#     --stochastic_round \
#     --stochastic_round_state \
#     --name new1_llama_2e-6_80_halfpre > new1_llama_2e-6_80_halfpre

# log文件的作用会被 > 后面的文件取代
# dtype原为bfloat16，现暂时改为float32