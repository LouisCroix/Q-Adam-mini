#!/bin/bash
#SBATCH -J q-adam-mini
#SBATCH -A L00120230003
#SBATCH -p p-A800
#SBATCH -N 1
#SBATCH -c 4
#SBATCH --gres=gpu:1


python3 ../Fine-Tuning-LLaMA-2-on-GSM8K/generate_eval.py > gsm_adamw_lora_2e-5_128_256_eval
