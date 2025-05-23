wandb login --relogin ade2fe7660f00be5a198e4af70789212e11eaae7

torchrun --standalone --nproc_per_node 1 ../my_run_pretrain_halfpre.py \
    --model_config ../configs/llama_1b.json \
    --lr 3e-4 \
    --batch_size 8 \
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
    --weight_group_size 256 \
    --stochastic_round \
    --name normal_32+8_no-sto > normal_32+8_no-sto
    
# log文件的作用会被 > 后面的文件取代
# dtype原为bfloat16，现暂时改为float32