wandb login --relogin 5bf505286321dfd08e945d35e78c7ad830bec99e

python ../my_run_finetune_lora.py \
    --model /hanyizhou/quant_adam_mini/models/models--meta-llama--Meta-Llama-3-8B/snapshots/62bd457b6fe961a42a631306577e622c83876cb6 \
    --task mmlu \
    --lr 4e-5 \
    --batch_size 1 \
    --total_batch_size 80 \
    --max_length 2048 \
    --num_epochs 3 \
    --weight_decay 0 \
    --lora_alpha 128 \
    --lora_rank 256 \
    --dtype float32 \
    --eval_every 50 \
    --optimizer adamw \
    --project 'finetune' \
    --name AdamW-llama_lora_4e-5_80_128_256_halfpre > AdamW_llama_lora_4e-5_80_128_256_halfpre_1
    
# log文件的作用会被 > 后面的文件取代
# dtype原为bfloat16，现暂时改为float32
# batch_size原为2，现暂时改为1（仅mmlu）
# lr原为2e-5，现暂时改为4e-5（仅mmlu）