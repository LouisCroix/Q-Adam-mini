wandb login --relogin ade2fe7660f00be5a198e4af70789212e11eaae7

torchrun --standalone --nproc_per_node 1 my_run_pretrain_halfpre.py \
    --model_config configs/llama_1b.json \
    --continue_from /mntcephfs/data/ruoyusun/hanyizhou/q-adam-mini-checkpoints/model_new1_1B_3e-4_256_halfpre_47500 \
    --lr 3e-4 \
    --batch_size 4 \
    --total_batch_size 256 \
    --max_length 1024 \
    --num_billion_training_tokens 1.5 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype float32 \
    --eval_every 500 \
    --save_every 5000 \
    --single_gpu \
    --optimizer q_adam_mini_8bit \
    --project '1B-weight_test' \
    --weight_quant \
    --weight_group_size 256 \
    --stochastic_round \
    --stochastic_round_state \
    --name new1_1B_3e-4_256_halfpre > new1_1B_3e-4_256_halfpre_cont4
    
# log文件的作用会被 > 后面的文件取代
# dtype原为bfloat16，现暂时改为float32