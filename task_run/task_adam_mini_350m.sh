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
    --model_config configs/llama_350m.json \
    --lr 1e-3 \
    --batch_size 16 \
    --total_batch_size 256 \
    --max_length 1024 \
    --num_billion_training_tokens 6.4 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype float32 \
    --eval_every 1000 \
    --save_every 350000 \
    --single_gpu \
    --optimizer adammini \
    --project '350M' \
    --name Adam_mini_new1-350M_1e-3_256_halfpre > Adam_mini_new1_350M_1e-3_256_halfpre
    
# log文件的作用会被 > 后面的文件取代
# dtype原为bfloat16，现暂时改为float32