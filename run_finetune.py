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
from trl import DataCollatorForCompletionOnlyLM

import datasets
import datasets.distributed
import evaluate
import wandb
# import deepspeed
from accelerate import Accelerator

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.llama_modeling import LlamaForCausalLM
from peft_pretraining.dataset import PreprocessedIterableDataset

import bitsandbytes as bnb
from q_adam_mini_8bit_dp import QAdamMini8bit   # _dp means suitable for deepspeed
from q_adamw_8bit_dp import QAdamW8bit
from adam_mini_dp import Adam_mini
from quantization import prepare_model_for_int8_training, QLinear
# from q_adam_mini_8bit_simulate import QGaLoreAdamW8bit_simulate
# from simulate_quantization import prepare_model_for_int8_training_simulation, SimQLinear
from setup import saving_model_weight, load_model_weight

transformers.logging.set_verbosity_error()

def parse_args(args):
    parser = argparse.ArgumentParser()
    
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
def evaluate_model(accelerator, model, tokenizer, pad_idx, world_size, batch_size, part="validation"):
    assert part in ["validation", "test"], "part of dataset for this function to use must be validation or test"
    _time = time.time()
    val_data = datasets.load_from_disk(f"./datasets/mmlu/all/{part}")
    val_data = val_data.shuffle(seed=42)
    metric = evaluate.load("accuracy")
    logger.info(f"Loaded {part} dataset and metric in {time.time() - _time:.2f} seconds")

    partial_prompt_gene = functools.partial(prompt_generator, tokenizer=tokenizer, max_length=args.max_length)
    val_dataset = val_data.map(partial_prompt_gene, batched=True, batch_size=1000, remove_columns=val_data.column_names)

    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.workers,
                                             collate_fn=DataCollatorForCompletionOnlyLM("<answer>: ", mlm=False, tokenizer=tokenizer, return_tensors="pt"))
    dataloader = accelerator.prepare(dataloader)

    evaluated_on_tokens = 0
    total_loss = 0
    total_batches = 1
    choice_ids = [tokenizer.convert_tokens_to_ids(f"{i}") for i in range(4)]
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")
    
    model.eval()

    with torch.no_grad():
        for batch in tqdm(dataloader):
            
            model_result = model(**batch)
            total_loss += model_result.loss

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size
            predictions = model_result.logits[(batch["labels"] != -100)][:, 15: 19].argmax(dim=1) + 15
            references = batch["labels"][batch["labels"] != -100]
            
            assert len(predictions) == len(references), f"predictions: {predictions}, references: {references}"
            references, predictions = accelerator.gather_for_metrics((references, predictions))
            metric.add_batch(references=references, predictions=predictions)
            
            total_batches += 1
            
    model.train()
    
    total_loss = total_loss / total_batches
    metric_result = metric.compute()["accuracy"]
    
    return total_loss, metric_result, evaluated_on_tokens


def main(args):
    set_seed(args.seed)

    accelerator = Accelerator(step_scheduler_with_optimizer=False)  # for deepspeed stage zero-3, put the optimizer initialization before accelerator initialization, 
                                                                    # since accelerator initialization with zero-3 breaks model stucture
    
    global_rank = int(os.environ['RANK'])
    world_size = int(os.environ["WORLD_SIZE"])
    assert world_size == torch.cuda.device_count(),  f"{world_size} {torch.cuda.device_count()}"

    logger.info("Process group initialized")

    # args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # from here, turn off logger if it is not in the main training process
    if not accelerator.is_main_process: logger.remove()

    # initialize wandb without config (it is passed later) in the main training process.
    if accelerator.is_main_process:
        if not args.unset_wandb:
            # os.environ["WANDB_MODE"] = "offline"    # comment this line if use wandb
            print("Initializing wandb")
            if args.wandb_api_key:
                wandb.login(key=args.wandb_api_key)
            wandb.init(project=args.project, name=args.name)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)
    
    # tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, max_length=args.max_length)
    tokenizer.add_special_tokens({"pad_token": "<<PAD>>"})

    if args.task == "mmlu":
        data = datasets.load_from_disk(f"./datasets/mmlu/all/auxiliary_train")
        partial_prompt_gene = functools.partial(prompt_generator, tokenizer=tokenizer, max_length=args.max_length)
    elif args.task == "gsm":
        data = datasets.load_from_disk(f"./datasets/gsm8k/main/train")
        partial_prompt_gene = functools.partial(gsm_prompt_generator, tokenizer=tokenizer, max_length=args.max_length)
    seed_for_shuffle = 42 

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    dataset = data.map(partial_prompt_gene, batched=True, batch_size=1000, remove_columns=data.column_names)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             collate_fn=DataCollatorForCompletionOnlyLM("<answer>: " if args.task == "mmlu" else "<answer>:", mlm=False, tokenizer=tokenizer, return_tensors="pt")) # batch_size和batch_sampler都是None，则直接返回dataset中的每一项
    epoch_traning_steps = math.ceil(len(dataloader)/(args.gradient_accumulation*world_size))
    args.num_training_steps = epoch_traning_steps*args.num_epochs

    model = HF_LlamaForCausalLM.from_pretrained(args.model)
    model.resize_token_embeddings(len(tokenizer), pad_to_multiple_of=4)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
        
    if args.weight_quant:
        # Enable INT8 training
        assert args.optimizer.lower() in ['q_adam_mini_8bit', 'q_adam_mini_8bit_per_layer', 'adamw8bit', 'adammini']
        target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        if args.simulation:
            pass
            # model = prepare_model_for_int8_training_simulation(model, args, target_module)
        else:
            model = prepare_model_for_int8_training(model, args, target_module)
        print('--'*20)
        print('Prepare Model for Int8 Training')
        print('--'*20)

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    
    # if args.dtype in ["bf16", "bfloat16"]:
    #     model = model.to(dtype=torch.bfloat16)

    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        load_model_weight(model, checkpoint_path)
        logger.info(f"Model successfully loaded (strict=False policy)")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}")
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(f"Will train for {args.num_training_steps - update_step} update steps")
        else:
            logger.warning(f"Did not find training state in {args.continue_from}, global step will start from zero")
        logger.info("*" * 40)
        
    print(f"Model parameter dtype: {model.model.layers[0].self_attn.q_proj.weight.data.dtype}\n")

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params_float = [p for p in model.parameters() if p.requires_grad]
    trainable_params_int8 = [p for p in model.parameters() if hasattr(p, 'group_size')]

    # Initialize config of wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'wiki103',
        "model": model.config.to_dict(),
        "world_size": world_size,
    })

    if accelerator.is_main_process:
        if not args.unset_wandb:
            wandb.config.update(run_config, allow_val_change=True)
            wandb.save(os.path.abspath(__file__), policy="now") # save current script

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    num_train_params = sum(p.numel() for p in trainable_params_float) + sum(p.numel() for p in trainable_params_int8)

    logger.info(f"Trainable params: {num_train_params / 1_000_000:.2f}M")
    if 'q_adam_mini' in args.optimizer.lower():
        logger.info(f"Trainable quantized params: {sum(p.numel() for p in trainable_params_int8) / 1_000_000:.2f}M")
    # logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")


    # prepare optimizer and scheduler according to command line arguments
    layer_wise_flag = False
    # Baseline optimizers
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params_float, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params_float, lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "sgd":
        optimizer = torch.optim.SGD(trainable_params_float, lr=args.lr, weight_decay=args.weight_decay, momentum=args.beta1)
    elif args.optimizer.lower() == "adafactor":
        args.beta1 = None if args.beta1 == 0.0 else args.beta1
        optimizer = transformers.optimization.Adafactor(
            trainable_params_float,
            lr=args.lr,
            eps=(1e-30, 1e-3),
            clip_threshold=1.0,
            decay_rate=-0.8,
            beta1=args.beta1,
            weight_decay=args.weight_decay,
            relative_step=False,
            scale_parameter=False,
            warmup_init=False,
        )
    elif args.optimizer.lower() == "adamw8bit":
        # optimizer = bnb.optim.AdamW8bit(trainable_params_float, lr=args.lr, weight_decay=args.weight_decay)
        optimizer = QAdamW8bit(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.optimizer.lower() == "adammini":
        optimizer = Adam_mini(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2), dim=model.config.hidden_size, n_heads=model.config.num_attention_heads)
    elif args.optimizer.lower() == "q_adam_mini_8bit":
        if args.simulation:
            pass
            # print('Using Simulation Mode')
            # optimizer = QGaLoreAdamW8bit_simulate(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2))
        else:
            optimizer = QAdamMini8bit(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2), dim=model.config.hidden_size, n_heads=model.config.num_attention_heads, stochastic_round_state=args.stochastic_round_state, otherwise_dtype=args.use_dtype)

    # Layer-wise optimizers
    elif args.optimizer.lower() == 'q_adam_mini_8bit_per_layer':
        optimizer_dict = {}
        for pname, p in model.named_parameters():
            if hasattr(p, 'group_size') or p.requires_grad:
                optimizer_dict[p] = QAdamMini8bit([(pname, p)], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2), dim=model.config.hidden_size, n_heads=model.config.num_attention_heads, stochastic_round_state=args.stochastic_round_state, otherwise_dtype=args.use_dtype)

        # get scheduler dict
        scheduler_dict = {}
        for p in model.parameters():
            if hasattr(p, 'group_size') or p.requires_grad:
                scheduler_dict[p] = training_utils.get_scheculer(
                    optimizer=optimizer_dict[p],
                    scheduler_type=args.scheduler,
                    num_training_steps=args.num_training_steps * args.gradient_accumulation,
                    warmup_steps=args.warmup_steps * args.gradient_accumulation,
                    min_lr_ratio=args.min_lr_ratio,
                )

        def optimizer_hook(p):
            if (not hasattr(p, 'float_grad')) and p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()
            scheduler_dict[p].step()

        # Register the hook onto every parameter block (tensor) using iteration through model.parameters()
        for p in model.parameters():
            if hasattr(p, 'group_size'):
                # suboptimal: backward_hook can not be applied to int8 tensors
                # we manully fuse the backward_hook inside the backward process
                setattr(p, 'backward_hook', optimizer_hook)
            elif p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    # scheduler
    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )
        
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading optimizer from {args.continue_from}")
        optimizer_checkpoint = torch.load(f"{args.continue_from}/optimizer.pt", map_location="cpu")
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info("*" * 40)
    
    if accelerator.is_main_process:
        print(f"Memory allocated 1 on {model.device}: {torch.cuda.memory_allocated()//(1024**2)}", flush=True)
        print(f"Memory reserved 1 on {model.device}: {torch.cuda.memory_reserved()//(1024**2)}", flush=True)
    
    model, optimizer, dataloader, scheduler = accelerator.prepare(model, optimizer, dataloader, scheduler)

    if accelerator.is_main_process:
        print(f"Memory allocated 2 on {model.device}: {torch.cuda.memory_allocated()//(1024**2)}", flush=True)
        print(f"Memory reserved 2 on {model.device}: {torch.cuda.memory_reserved()//(1024**2)}", flush=True)

    for name, module in model.named_modules():
        if isinstance(module, QLinear):
            print(module.weight.device, module.weight.scales.device)
            weight_device = module.weight.device
            module.weight.scales = module.weight.scales.to(device=weight_device)
            module.weight.zeros = module.weight.zeros.to(device=weight_device)

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    epoch = 0
    period_loss = 0

    # ##############################
    # TRAINING LOOP
    # ##############################

    for epoch in range(args.num_epochs):
        
        # train
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        if accelerator.is_main_process: pbar = tqdm(total=epoch_traning_steps, desc="Update steps per epoch", ncols=80)
        epoch_step = 0
        
        for batch_idx, batch in enumerate(dataloader):

            global_step += 1
            local_step += 1
            epoch_step += 1

            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
            is_accumulating = epoch_step % args.gradient_accumulation != 0 and global_step % len(dataloader) != 0

            with accelerator.accumulate(model):
                loss = model(**batch).loss
                
                accelerator.backward(loss)  # accelerator scale the loss by itself when using gradient accumulation
                period_loss += loss.item() / args.gradient_accumulation

            if is_accumulating:
                continue

            if update_step%(math.ceil(args.num_training_steps/300)) == 0 and accelerator.is_main_process:
                if update_step > 0:
                    print(f"\nUpdate step: {update_step}. Tokens seen: {tokens_seen}")
                    print(f"Training loss: {period_loss/(math.ceil(args.num_training_steps/300))}")
                    if not layer_wise_flag:
                        lr = optimizer.param_groups[0]["lr"]
                    else:
                        lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
                    print(f"Learning rate: {lr}")
                period_loss = 0
                print(f"Memory allocated: {torch.cuda.memory_allocated()//(1024**2)}", flush=True)
                print(f"Memory reserved: {torch.cuda.memory_reserved()//(1024**2)}", flush=True)

            # The below code is only executed during the update step
            # add grad norm clipping (a kind of grad normalization to make grad norm less or equal to the clipping limit)
            if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params_float, args.grad_clipping)

            if accelerator.is_main_process: pbar.update(1)
            
            if not layer_wise_flag:
                # model_engine.step()
                # scaler.step(optimizer)
                # scaler.update()
                optimizer.step()
                scheduler.step()        # after the last run scheduler.step(), the lr will become its original value, which is not used by the optimizer.
                optimizer.zero_grad()

            update_step += 1
            update_time = time.time() - update_time
            
            # evaluation
            if args.task == "mmlu":
                accelerator.wait_for_everyone()
                if update_step % args.eval_every == 0 or update_step == args.num_training_steps:
                    logger.info(f"Performing evaluation at epoch {epoch} step {update_step}")
                    total_loss, accuracy, evaluated_on_tokens = evaluate_model(
                        accelerator, model, tokenizer, pad_idx,  world_size, args.batch_size
                    )
                    if accelerator.is_main_process:
                        if not args.unset_wandb:
                            wandb.log({
                                "final_eval_loss": total_loss,
                                "eval_accuracy": accuracy,
                                "final_eval_tokens": evaluated_on_tokens,
                                },
                                step=global_step,
                            )
                    logger.info(f"Eval loss at step {update_step}: {total_loss}, accuracy: {accuracy}")

            if not layer_wise_flag:
                lr = optimizer.param_groups[0]["lr"]
            else:
                lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size

            if accelerator.is_main_process:
                if not args.unset_wandb:
                    wandb.log({
                        "loss": loss.item(),
                        "lr": lr,
                        "update_step": update_step,
                        "tokens_seen": tokens_seen,
                        "throughput_tokens": tokens_in_update / update_time,
                        "throughput_examples": args.total_batch_size / update_time,
                        "throughput_batches": batches_in_update / update_time,
                        },
                        step=global_step,
                    )
            update_time = time.time()
        
        if accelerator.is_main_process: pbar.close()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    print(f"Rank {global_rank} stopping training.")

    # Final evaluation
    if args.task == "mmlu":
        logger.info("Running final test")
        accelerator.wait_for_everyone()

        torch.cuda.empty_cache()

        total_loss, accuracy, evaluated_on_tokens = evaluate_model(
            accelerator, model, tokenizer, pad_idx, world_size, args.batch_size, "test"
        )

        if accelerator.is_main_process:
            if not args.unset_wandb:
                wandb.log({
                    "final_test_loss": total_loss,
                    "test_accuracy": accuracy,
                    "final_test_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(f"Final test loss: {total_loss}, accuracy: {accuracy}")
        
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        current_model_directory = f"{args.save_dir}/model_{args.name}"
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(current_model_directory, exist_ok=True)
        unwrapped_model = accelerator.unwrap_model(model)
        unwrapped_model.save_pretrained(current_model_directory, save_function=accelerator.save)

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)