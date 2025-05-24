'''
主要改动：
import的包换成从q_adam_mini_8bit导入，删除导入其他自定义的包的代码。
删除和其他自定义包有关的代码。
为继续训练增加读取之前的optimizer的操作（直接存取optimizer.state_dict即可，我的optimizer实现已经考虑到了Adam-mini新增的优化信息的保存问题）（有需要再进行，大概率不需要，不过用到时要注意，需要读取和保存的有optimizer和optimizer_dict两种可能）
'''
import os
import re
import time
import json
import math
import random
import argparse
import numpy as np
import functools

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
# from torch.amp import autocast, GradScaler

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, set_seed
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM
from transformers.integrations import WandbCallback
from trl import DataCollatorForCompletionOnlyLM, SFTConfig, SFTTrainer # transformers的DataCollator负责pad但不负责truncation，trl的也一样
from peft import LoraConfig

import datasets
import datasets.distributed
import evaluate
import wandb
# import deepspeed

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.llama_modeling import LlamaForCausalLM
from peft_pretraining.dataset import PreprocessedIterableDataset

transformers.logging.set_verbosity_info()

def parse_args(args):
    parser = argparse.ArgumentParser()
    
    # for deepspeed use
    # parser.add_argument("--local_rank", type=int, default=1)    # the value of --local_rank is automatically assigned
    # parser = deepspeed.add_config_arguments(parser)             # two args added by deepspeed here, --deepspeed and --deepspeed_config

    # training parameters
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--task", type=str, choices=["mmlu", "gsm"], required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--warmup_steps", type=int, default=0)
    parser.add_argument("--eval_every", type=int, default=50)

    parser.add_argument("--save_dir", type=str, default="./q-adam-mini-checkpoints")
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--project", type=str, default="test")
    parser.add_argument("--unset_wandb", action="store_true")
    parser.add_argument("--wandb_api_key", type=str, default=None, help="API key for wandb login")
    parser.add_argument("--grad_clipping", type=float, default=0.0)

    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)

    # beta2 for AdamW
    parser.add_argument("--beta2", type=float, default=0.95)

    # Q-Adam-mini parameters
    parser.add_argument("--weight_quant", action='store_true')
    parser.add_argument("--weight_bits", type=int, default=8)
    parser.add_argument("--weight_group_size", type=int, default=256)
    parser.add_argument("--stochastic_round", action='store_true')          # for param update
    parser.add_argument("--stochastic_round_state", action='store_true')    # for optimizer state opdate
    parser.add_argument("--simulation", action='store_true')
    # Current weight quantization implementation does not support DDP
    
    # Lora parameters
    parser.add_argument("--lora_rank", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=128)

    # disable ddp, single_gpu
    # parser.add_argument("--single_gpu", default=False, action="store_true")

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    assert args.task in ["mmlu", "gsm"], "argument dataset should be mmlu or gsm"
    return args

def str2int(s):
    return int(s.replace(',', ''))

def prompt_generator(batch, tokenizer, max_length):
    input_ids = []
    for i in range(len(batch["question"])):
        text = f'<question>: {batch["question"][i]}\n<choices>: {batch["choices"][i]}\n<answer>: {batch["answer"][i]}'
        input_ids.append(tokenizer(text, max_length=max_length, truncation=True)["input_ids"])
        assert type(input_ids[-1]) == list, type(input_ids[-1])
    return {"input_ids": input_ids}

def gsm_prompt_generator(batch, tokenizer, max_length):
    input_ids = []
    for i in range(len(batch["question"])):
        text = f'<question>: {batch["question"][i]}\n<answer>: {batch["answer"][i]}'
        input_ids.append(tokenizer(text, max_length=max_length, truncation=True)["input_ids"])
        # print(tokenizer(text, max_length=max_length, truncation=True)["input_ids"], tokenizer("<answer>: "))
        assert type(input_ids[-1]) == list, type(input_ids[-1])
    return {"input_ids": input_ids}

def gsm_num_prompt_generator(batch, tokenizer, max_length):
    input_ids, num_answers = [], []
    
    for i in range(len(batch["question"])):
        assert "#### " in batch["answer"][i], batch["answer"][i]
        num_answer = str2int(batch["answer"][i].split("#### ")[-1])
        num_answers.append(num_answer)
        
        text = f'<question>: {batch["question"][i]}\n<answer>: '
        input_ids.append(tokenizer(text, max_length=max_length, truncation=True)["input_ids"])
        assert type(input_ids[-1]) == list, type(input_ids[-1])
        
    return {"input_ids": input_ids, "num_answers": num_answers}

@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, batch_size, part="validation"):
    assert part in ["validation", "test"], "part of dataset for this function to use must be validation or test"
    _time = time.time()

    val_data = datasets.load_from_disk(f"./datasets/mmlu/all/{part}")

    val_data = val_data.shuffle(seed=42)
    metric = evaluate.load("./evaluate-main/evaluate-main/metrics/accuracy")
    logger.info(f"Loaded {part} dataset and metric in {time.time() - _time:.2f} seconds")

    partial_prompt_gene = functools.partial(prompt_generator, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = val_data.map(partial_prompt_gene, batched=True, batch_size=1000, remove_columns=val_data.column_names)
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.workers,
                                             collate_fn=DataCollatorForCompletionOnlyLM("<answer>: ", mlm=False, tokenizer=tokenizer, return_tensors="pt"))

    evaluated_on_tokens = 0
    total_loss = 0
    total_batches = 1
    choice_ids = [tokenizer.convert_tokens_to_ids(f"{i}") for i in range(4)]
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")
    
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):

            batch = {k: v.to(model.device) for k, v in batch.items()}
            
            model_result = model(**batch)
            total_loss += model_result.loss

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item()
            predictions = model_result.logits[(batch["labels"] != -100)][:, 15: 19].argmax(dim=1) + 15
            references = batch["labels"][batch["labels"] != -100]
            
            assert len(predictions) == len(references), f"predictions: {predictions}, references: {references}"
            metric.add_batch(references=references, predictions=predictions)
            
            total_batches += 1
            
    model.train()
    
    total_loss = total_loss / total_batches
    metric_result = metric.compute()["accuracy"]
    
    return total_loss, metric_result, evaluated_on_tokens


def main(args):
    # set wandb environ for huggingface trainer
    os.environ["WANDB_PROJECT"] = args.project
    os.environ["WANDB_LOG_MODEL"] = "false"
    os.environ["WANDB_WATCH"] = "false"

    logger.info("Process group initialized")

    # args.gradient_accumulation * args.batch_size == args.total_batch_size
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            args.gradient_accumulation = args.total_batch_size // args.batch_size
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size == args.total_batch_size, \
        "gradient_accumulation * batch_size must be equal to total_batch_size"

    # initialize wandb without config (it is passed later) in the main training process. wandb是用于可视化监测训练过程的库
    if not args.unset_wandb:
        # os.environ["WANDB_MODE"] = "offline"    # comment this line if use online wandb
        print("Initializing wandb")
        if args.wandb_api_key:
            wandb.login(key=args.wandb_api_key)

    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=args.max_length)
    tokenizer.add_special_tokens({"pad_token": "<<PAD>>"})
    # tokenizer.add_special_tokens({"additional_special_tokens": ["<question>", "<answer>"]})

    if args.task == "mmlu":
        val_data = datasets.load_from_disk(f"./datasets/mmlu/all/validation")
        partial_prompt_gene = functools.partial(prompt_generator, tokenizer=tokenizer, max_length=args.max_length)
        val_dataset = val_data.map(partial_prompt_gene, batched=True, batch_size=1000, remove_columns=val_data.column_names)
        
        data = datasets.load_from_disk(f"./datasets/mmlu/all/auxiliary_train")
        partial_prompt_gene = functools.partial(prompt_generator, tokenizer=tokenizer, max_length=args.max_length)
        
    elif args.task == "gsm":
        data = datasets.load_from_disk(f"./datasets/gsm8k/main/train")
        partial_prompt_gene = functools.partial(gsm_prompt_generator, tokenizer=tokenizer, max_length=args.max_length)

    dataset = data.map(partial_prompt_gene, batched=True, batch_size=1000, remove_columns=data.column_names)

    model = HF_LlamaForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4)
    
    epoch_traning_steps = math.ceil(len(dataset)/(args.gradient_accumulation*args.batch_size))
    args.num_training_steps = epoch_traning_steps*args.num_epochs
    print(epoch_traning_steps, args.num_training_steps)
    
    metric = evaluate.load("./evaluate-main/evaluate-main/metrics/accuracy")
    
    def compute_metrics(eval_pred, compute_result):
        logits, labels = eval_pred
        predictions = logits[labels != -100][:, 15: 19].argmax(dim=1) + 15
        references = labels[labels != -100]
        metric.add_batch(references=references, predictions=predictions)
        if compute_result:
            return metric.compute()

    
    class MyCallback(WandbCallback):

        def __init__(self, gradient_accumulation=args.gradient_accumulation):
            super().__init__()
            self.gradient_accumulation = gradient_accumulation
            self.prev_eval_step = 0

        def on_log(self, args, state, control, model=None, logs=None, **kwargs):
            if state.is_world_process_zero and "loss" in logs:
                print(state.global_step, flush=True)
                wandb.log({
                    "loss": logs['loss'],
                    "lr": logs['learning_rate'],
                    "update_step": state.global_step,
                    },
                    step=state.global_step*self.gradient_accumulation,
                )
                
        def on_evaluate(self, args, state, control, **kwargs):
            if state.is_world_process_zero:
                print(state.log_history[-1], flush=True)
                logs = state.log_history[-1]
                if self.prev_eval_step != state.global_step:    # validation
                    self.prev_eval_step = state.global_step
                    wandb.log({
                        "final_eval_loss": logs['eval_loss'],
                        "eval_accuracy": logs['eval_accuracy'],
                        },
                        step=state.global_step*self.gradient_accumulation,
                        # commit=False,
                    )
                else:                                           # test
                    wandb.log({
                        "final_test_loss": logs['eval_loss'],
                        "test_accuracy": logs['eval_accuracy'],
                        },
                        step=state.global_step*self.gradient_accumulation,
                        # commit=True
                    )


    # ##############################
    # TRAINING
    # ##############################
    
    peft_config = LoraConfig(
        lora_alpha=args.lora_rank,          # Q-GaLore
        lora_dropout=0.05,      # Q-GaLore
        r=args.lora_rank,                  # Adam-mini
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head"],
    )

    # Step 4: Training Arguments
    training_args = SFTConfig(
        seed=args.seed,
        data_seed=42,
        output_dir=f"{args.save_dir}/model_{args.name}",
        max_steps=args.num_training_steps,
        max_length=args.max_length,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        # eval_accumulation_steps=10,
        batch_eval_metrics=True,
        optim="adamw_torch",
        adam_beta1=0.9,
        adam_beta2=args.beta2,
        weight_decay=args.weight_decay,
        save_strategy="no",
        learning_rate=args.lr,
        fp16=False,
        bf16=True,
        max_grad_norm=None,
        lr_scheduler_type="cosine_with_min_lr",
        lr_scheduler_kwargs={"min_lr_rate": 0.1},
        eval_strategy="steps" if args.task == "mmlu" else "no",
        eval_steps=args.eval_every,
        logging_strategy="steps",
        logging_steps=1,
        report_to="none",
        run_name=args.name,
    )

    # Initialize the Trainer and start fine-tuning
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        eval_dataset=val_dataset if args.task == "mmlu" else None,
        peft_config=peft_config,
        # tokenizer=tokenizer,
        data_collator=DataCollatorForCompletionOnlyLM("<answer>: " if args.task == "mmlu" else "<answer>:", mlm=False, tokenizer=tokenizer, return_tensors="pt"),
        compute_metrics=compute_metrics,
        callbacks=[MyCallback()],
        args=training_args,
    )

    trainer.train()

    # ##############################
    # END of training
    # ##############################
    logger.info("Training finished")

    # Final evaluation and test
    if args.task == "mmlu":
        logger.info("Running final eval and test")

        torch.cuda.empty_cache()
        
        trainer.evaluate()  # also triggers wandbcallback
        
        test_data = datasets.load_from_disk(f"./datasets/mmlu/all/test")
        test_data = test_data.shuffle(seed=42)
        partial_prompt_gene = functools.partial(prompt_generator, tokenizer=tokenizer, max_length=args.max_length)
        test_dataset = test_data.map(partial_prompt_gene, batched=True, batch_size=1000, remove_columns=test_data.column_names)
        trainer.evaluate(test_dataset)

    logger.info("Script finished successfully")
    print(f"Script finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)