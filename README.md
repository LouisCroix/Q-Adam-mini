

# Q-Adam-mini

This repo contains the pre-release version of Q-Adam-mini algorithm.

We propose Q-Adam-mini, a memory-efficient optimizer for Large Language Model (LLM) training that achieves 8× reduction in GPU memory usage while maintaining performance parity with full-precision AdamW. Building upon Adammini (Zhang et al., 2024b), which achieves 50% memory savings over AdamW, we further optimize memory efficiency by quantizing optimizer states. We achieve this by: (i) quantizing the firstorder momentum (m) to INT8 and (ii) retaining the second-order momentum (v) in FP32, which occupies less than 1% of total memory. However, a weight norm explosion problem occurs in the embedding layer. We analyze this issue and address it by applying stochastic rounding for momentum quantization exclusively to the embedding layer. We validate our approach on both pre-training and fine-tuning tasks, with the model size ranging from 60M to 8B. Our results demonstrate that Q-Adam-mini enables scalable LLM training with limited computational resources.



## Install Q-Adam-mini

##### install from pip

```bash
pip install -r requirements.txt
```





### Usage

##### Pretraining model on C4 dataset

```bash
torchrun --standalone --nproc_per_node 1 my_run_pretrain_halfpre.py \
    --model_config configs/llama_7b.json \
    --lr 3e-4 \
    --batch_size 8 \
    --total_batch_size 256 \
    --max_length 32 \
    --num_billion_training_tokens 0.1 \
    --warmup_steps 2000 \
    --weight_decay 0.1 \
    --dtype float32 \
    --eval_every 500 \
    --save_every 5000 \
    --single_gpu \
    --optimizer q_adam_mini_8bit \
    --project '7B-pretrain' \
    --weight_quant \
    --weight_group_size 256 \
    --stochastic_round \
    --stochastic_round_state \
```



##### Fine-tuning Llama3-8B

```bash
torchrun --standalone --nproc_per_node 1 my_run_finetune.py \
    --model Your/Model/Path \
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
```



