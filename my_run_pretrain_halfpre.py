'''
主要改动：
import的包换成从q_adam_mini_8bit导入，删除导入其他自定义的包的代码。
删除和其他自定义包有关的代码。
为继续训练增加读取之前的optimizer的操作（直接存取optimizer.state_dict即可，我的optimizer实现已经考虑到了Adam-mini新增的优化信息的保存问题）（有需要再进行，大概率不需要，不过用到时要注意，需要读取和保存的有optimizer和optimizer_dict两种可能）
'''
import os
import time
import json
import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.amp import autocast, GradScaler

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM, DataCollatorForLanguageModeling, set_seed
from transformers import LlamaForCausalLM as HF_LlamaForCausalLM

import datasets
import datasets.distributed
import wandb

from tqdm import tqdm
from loguru import logger

from peft_pretraining import training_utils, args_utils
from peft_pretraining.llama_modeling import LlamaForCausalLM
from peft_pretraining.dataset import PreprocessedIterableDataset

import bitsandbytes as bnb
from q_adam_mini_8bit_new_32 import QAdamMini8bit
from adam_mini_new import Adam_mini
from quantization import prepare_model_for_int8_training, QLinear
# from q_adam_mini_8bit_simulate import QGaLoreAdamW8bit_simulate
# from simulate_quantization import prepare_model_for_int8_training_simulation, SimQLinear
from setup import saving_model_weight, load_model_weight

transformers.logging.set_verbosity_error()  # 让transformers日志只记录错误和更高级别的信息

def parse_args(args):
    parser = argparse.ArgumentParser()

    # training parameters
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--use_hf_model", default=False, action="store_true")   # action="store_true"表示该参数若在命令行中出现则被设置为True，否则为default参数，default参数未提供则为False（没有设置action时default参数的默认值是None）
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=None)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--optimizer", default="Adam")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler", type=str, default="cosine", choices=["linear", "cosine", "cosine_restarts"])
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)  # warm-up后最低学习率占最高学习率的比例
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=500)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument("--num_billion_training_tokens", type=float, default=1.1)
    parser.add_argument("--num_training_steps", type=int, default=None,
                        help="Number of **update steps** to train for. "
                             "Notice that gradient accumulation is taken into account.")
    parser.add_argument("--max_train_tokens", type=training_utils.max_train_tokens_to_number, default=None,
                        help="Number of tokens to train on. Overwrites num_training_steps. "
                             "You can use M and B suffixes, e.g. 100M or 1B.")
    parser.add_argument("--save_every", type=int, default=5_000)
    parser.add_argument("--save_dir", type=str, default="./q-adam-mini-checkpoints")
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument("--name", type=str, default='test')
    parser.add_argument("--dtype", type=str, default="bfloat16" if torch.cuda.is_bf16_supported() else "float32")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)  # 为torch，numpy，和random库提供的随机种子，默认值为0
    parser.add_argument("--project", type=str, default="test")
    parser.add_argument("--unset_wandb", action="store_true")
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
    parser.add_argument("--single_gpu", default=False, action="store_true")

    args = parser.parse_args(args)
    args.num_training_tokens = int(args.num_billion_training_tokens*1e9)
    if args.num_training_steps == None:
        args.num_training_steps = round(args.num_training_tokens*1.2/(args.total_batch_size*args.max_length))

    args = args_utils.check_args_torchrun_main(args)    # 为torchrun检查参数并设置没有提供的参数值
    return args

# 加载验证数据集并计算模型的验证loss和用于验证的非padding token数（分布式训练时会乘以world_size来估算整体训练的数据，不使用dist.all_gather()可能是因为非padding token数需要在验证循环中计算来确定验证何时停止，多次计算需要保证效率）
@torch.no_grad()
def evaluate_model(model, tokenizer, pad_idx, global_rank, world_size, device, batch_size):
    _time = time.time()
    val_data = datasets.load_dataset("./datasets/c4_val", split="validation", streaming=True) #DGX
    val_data = val_data.shuffle(seed=42)
    logger.info(f"Loaded validation dataset in {time.time() - _time:.2f} seconds")

    if not args.single_gpu:
        val_data = datasets.distributed.split_dataset_by_node(val_data, rank=global_rank, world_size=world_size)

    # val_data_mapped = val_data.map(
    #     preprocess_batched,
    #     batched=True,
    #     remove_columns=["text", "timestamp", "url"],
    # )
    # val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(val_data_mapped, batch_size) # training_utils.batch_fn()用于将数据集转化为多批次，并逐批次转化为tensor

    val_dataset = PreprocessedIterableDataset(val_data, tokenizer, batch_size=batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, num_workers=args.workers,
                                             collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt"))

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")
    
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            if evaluated_on_tokens > target_eval_tokens:
                break
            total_batches += 1

            batch = {k: v.to(device) for k, v in batch.items()}
            # labels = batch["input_ids"].clone()
            # labels[labels == pad_idx] = -100
            loss = model(**batch).loss
            total_loss += loss

            evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size
        
    model.train()

    total_loss = total_loss / total_batches
    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens


def main(args):
    # 设置所有随机数种子
    # torch.manual_seed(args.seed)
    # np.random.seed(args.seed)
    # random.seed(args.seed)
    set_seed(args.seed) # 主程序中设置的seed在子程序中也适用

    # 分布式训练准备工作
    assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    global_rank = int(os.environ['RANK'])
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(local_rank)

    logger.info(f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}")

    dist.init_process_group(backend="nccl", rank=global_rank, world_size=world_size)

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    # 检查并完成梯度累积相关超参数的设置，公式：args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size
    if args.total_batch_size is not None:
        if args.gradient_accumulation is None:
            assert args.total_batch_size % world_size == 0, "total_batch_size must be divisible by world_size"
            args.gradient_accumulation = args.total_batch_size // (args.batch_size * world_size)
            assert args.gradient_accumulation > 0, "gradient_accumulation must be greater than 0"

    assert args.gradient_accumulation * args.batch_size * world_size == args.total_batch_size, \
        "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # from here, turn off logger if it is not in the main training process
    if global_rank != 0: logger.remove()

    # initialize wandb without config (it is passed later) in the main training process. wandb是用于可视化监测训练过程的库
    if global_rank == 0:
        if not args.unset_wandb:
            os.environ["WANDB_MODE"] = "offline"    # 若使用online wandb则注释掉此行
            print("Initializing wandb")
            wandb_id = None
            if args.continue_from is not None:
                with open(f"{args.continue_from}/wandb.json", "r") as f:
                    wandb_id = json.load(f)["wandb_id"]
            wandb.init(project=args.project, name=args.name, id=wandb_id, resume=args.continue_from is not None)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    # 数据加载并分布到不同设备
    data = datasets.load_dataset("/L00120230003/c4", split="train", streaming=True)
    seed_for_shuffle = 42 

    logger.info(f"Shuffling data with seed {seed_for_shuffle}")
    data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)
    if not args.single_gpu:
        data = datasets.distributed.split_dataset_by_node(
            data, rank=global_rank, world_size=world_size,
        )

    # it doesn't matter which tokenizer we use, because we train from scratch
    # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice
    tokenizer = AutoTokenizer.from_pretrained("./models/snapshots/01c7f73d771dfac7d292323805ebc428287df4f9", max_length=args.max_length)
    tokenizer.add_special_tokens({"pad_token": "<<PAD>>"})
    print(f"\npad_token_id: {tokenizer.pad_token_id}\n")
    
    # 数据批量映射函数，映射方式为将数据tokenize
    # def preprocess_batched(batch):
    #     batch = tokenizer(
    #         batch["text"],
    #         max_length=args.max_length,
    #         truncation=True,
    #         padding="max_length",
    #         return_tensors="pt",
    #     )
    #     return batch

    dataset = PreprocessedIterableDataset(data, tokenizer, batch_size=args.batch_size, max_length=args.max_length)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=args.workers,
                                             collate_fn=DataCollatorForLanguageModeling(tokenizer, mlm=False, return_tensors="pt")) # batch_size和batch_sampler都是None，则直接返回dataset中的每一项

    model_config = AutoConfig.from_pretrained(args.model_config)
    model_config.vocab_size += 1
    if args.use_hf_model:
        model: HF_LlamaForCausalLM = AutoModelForCausalLM.from_config(model_config)
    else:
        model = LlamaForCausalLM(model_config)

    if args.activation_checkpointing:
        model.gradient_checkpointing_enable()
    # print("Before quantization:", model.model.layers[0].self_attn.q_proj.weight.data)   # 检查随机种子是否设置成功

    if args.weight_quant:
        # Enable INT8 training
        assert args.optimizer.lower() in ['q_adam_mini_8bit', 'q_adam_mini_8bit_per_layer', 'adamw8bit', 'adammini']
        target_module = ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'up_proj', 'down_proj', 'gate_proj']
        # 对模型做量化或模拟量化处理，要处理的层名称包含在target_module中
        if args.simulation:
            pass
            # model = prepare_model_for_int8_training_simulation(model, args, target_module)
        else:
            model = prepare_model_for_int8_training(model, args, target_module)
        print('--'*20)
        print('Prepare Model for Int8 Training')
        print('--'*20)
    # print("After quantization:", model.model.layers[0].self_attn.q_proj.weight.data)

    global_step = 0
    update_step = 0
    beginning_step = 0
    tokens_seen = 0
    tokens_seen_before = 0
    
    # if args.dtype in ["bf16", "bfloat16"]:
    #     model = model.to(dtype=torch.bfloat16)  # pytorch模块的to()函数只接受浮点dtype，并且不会把整数参数转化为dtype

    # 若是继续训练，则载入之前的数据
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(f"Loading model from {args.continue_from}")
        checkpoint_path = os.path.join(args.continue_from, "pytorch_model.bin")
        load_model_weight(model, checkpoint_path)   # setup.py中定义的用于载入量化模型的函数
        # model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"), strict=True)
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
        
    print(f"Model parameters dtype: {model.model.layers[0].self_attn.q_proj.weight.data.dtype}\n")
        
    memory_before_model = torch.cuda.memory_allocated()//(1024**2)
    print(f"Memory allocated before model put to GPU: {torch.cuda.memory_allocated()//(1024**2)}, Unit: MB, same below.")
    
    model = model.to(device=device)
    
    memory_after_model = torch.cuda.memory_allocated()//(1024**2)
    print(f"Memory allocated after model put to GPU: {torch.cuda.memory_allocated()//(1024**2)}")
    print(f"Memory ocupied by model: {memory_after_model - memory_before_model}")

    # INT8 training: move the scales and zeros of all QLinear to the same device as the weight
    for name, module in model.named_modules():
        if isinstance(module, QLinear):
            weight_device = module.weight.device
            module.weight.scales = module.weight.scales.to(device=weight_device)
            module.weight.zeros = module.weight.zeros.to(device=weight_device)

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params_float = [p for p in model.parameters() if p.requires_grad]
    trainable_params_int8 = [p for p in model.parameters() if hasattr(p, 'group_size')] # 注意量化参数的float_grad属性是到backward之后才有的，所以这里用group_size属性判断是否为量化参数（其实一直用这个属性判断就可以，不用改成float_grad）

    # Initialize config of wandb
    run_config = dict(vars(args))
    run_config.update({
        "max_lr": run_config.pop("lr"),  # rename lr to max_lr to avoid conflicts with scheduler
        "total_params_M": n_total_params / 1_000_000,
        "dataset": 'wiki103',
        "model": model_config.to_dict(),
        "world_size": world_size,
        "device": str(device),
    })

    if global_rank == 0:
        if not args.unset_wandb:
            wandb.config.update(run_config, allow_val_change=True)
            wandb.save(os.path.abspath(__file__), policy="now") # save current script
        # fix tqdm visual length to 80 so that the progress bar
        # doesn't jump around when changing from external display to laptop
        pbar = tqdm(total=args.num_training_steps - update_step, desc="Update steps", ncols=80)

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M")
    num_train_params = sum(p.numel() for p in trainable_params_float) + sum(p.numel() for p in trainable_params_int8)

    logger.info(f"Trainable params: {num_train_params / 1_000_000:.2f}M")
    if 'q_adam_mini' in args.optimizer.lower():
        logger.info(f"Trainable quantized params: {sum(p.numel() for p in trainable_params_int8) / 1_000_000:.2f}M")
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")


    # prepare optimizer and scheduler according to command line arguments
    layer_wise_flag = False
    # Baseline optimizers
    if args.optimizer.lower() == "adam":
        optimizer = torch.optim.Adam(trainable_params_float, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2))
    elif args.optimizer.lower() == "adamw":
        optimizer = torch.optim.AdamW(trainable_params_float, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2))
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
        optimizer = bnb.optim.AdamW8bit(trainable_params_float, lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2))
    elif args.optimizer.lower() == "adammini":
        optimizer = Adam_mini(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2), dim=model_config.hidden_size, n_heads=model_config.num_attention_heads)
    elif args.optimizer.lower() == "q_adam_mini_8bit":
        if args.simulation:
            pass
            # print('Using Simulation Mode')
            # optimizer = QGaLoreAdamW8bit_simulate(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2))
        else:
            optimizer = QAdamMini8bit(model.named_parameters(), lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2), dim=model_config.hidden_size, n_heads=model_config.num_attention_heads, stochastic_round_state=args.stochastic_round_state, otherwise_dtype=args.use_dtype)

    # Layer-wise optimizers
    elif args.optimizer.lower() == 'q_adam_mini_8bit_per_layer':
        # TODO: seems scheduler call twice in one update step, need to check, for now double the num_training_steps, warmup_steps and update_proj_gap 这是LOMO不能accumulate gradient导致的
        optimizer_dict = {}
        for pname, p in model.named_parameters():
            if hasattr(p, 'group_size') or p.requires_grad:
                optimizer_dict[p] = QAdamMini8bit([(pname, p)], lr=args.lr, weight_decay=args.weight_decay, betas=(0.9, args.beta2), dim=model_config.hidden_size, n_heads=model_config.num_attention_heads, stochastic_round_state=args.stochastic_round_state, otherwise_dtype=args.use_dtype)

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

        # 注册LOMO hook
        def optimizer_hook(p):
            if (not hasattr(p, 'float_grad')) and p.grad is None: 
                return
            optimizer_dict[p].step()
            optimizer_dict[p].zero_grad()   # 注意：q-adam-mini的zero_grad()默认将grad设置为None，即会清除其内存占用
            scheduler_dict[p].step()

        # Register the hook onto every parameter block (tensor) using iteration through model.parameters() 因此本程序中LOMO方法的实现是依次更新每个tensor
        for p in model.parameters():
            if hasattr(p, 'group_size'):
                # suboptimal: backward_hook can not be applied to int8 tensors
                # we manully fuse the backward_hook inside the backward process 不是用官方函数注册，而是直接设置backward_hook属性的值为hook函数，这个属性会在W8Linear的backward执行的末尾被调用，参见quantization.py
                setattr(p, 'backward_hook', optimizer_hook)
            elif p.requires_grad:
                p.register_post_accumulate_grad_hook(optimizer_hook)

        layer_wise_flag = True

    else:
        raise ValueError(f"Optimizer {args.optimizer} not supported")

    # scheduler 前面非LOMO的情况没有定义scheduler，这里补上
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
        optimizer_checkpoint = torch.load(f"{args.continue_from}/optimizer.pt", map_location="cpu") # 东西不止optimizer和scheduler，不要一次性都弄到GPU上
        optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        logger.info("*" * 40)

    # prepare DDP
    if not args.single_gpu:
        model: HF_LlamaForCausalLM = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
        )

    scaler = GradScaler()

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step
    period_loss = 0

    # ##############################
    # TRAINING LOOP
    # we'll never go through all the data, so no need for epochs
    # ##############################

    while tokens_seen <= args.num_training_tokens:
        
        for batch_idx, batch in enumerate(dataloader):

            global_step += 1
            local_step += 1

            # 达到要求的training steps之后结束训练
            if tokens_seen > args.num_training_tokens:
                logger.info(f"Reached max number of training tokens ({args.num_training_tokens}) at update step {update_step}. Stopping training.")
                print(f"Rank {global_rank} stopping training.")
                break

            batch = {k: v.to(device) for k, v in batch.items()}
            # labels = batch["input_ids"].clone()
            # labels[labels == pad_idx] = -100
            tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size
            # if (batch["input_ids"] != pad_idx).sum().item() * world_size % args.max_length != 0: print(f'pad_tokens_seen: {(batch["input_ids"] == pad_idx).sum().item() * world_size}')

            with autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(**batch).loss

            scaled_loss = loss / args.gradient_accumulation
            # print("--------scaled_loss", scaled_loss.dtype)
            scaler.scale(scaled_loss).backward()
            period_loss += scaled_loss.item()

            # 达到梯度累积要求的steps之后再执行后序的梯度更新等操作
            if global_step % args.gradient_accumulation != 0:
                continue
            
            if update_step%(args.num_training_steps//500) == 0:
                if update_step > 0:
                    print(f"\nUpdate step: {update_step}. Tokens seen: {tokens_seen}")
                    print(f"Training loss: {period_loss/(args.num_training_steps//500)}")
                    if not layer_wise_flag:
                        lr = optimizer.param_groups[0]["lr"]
                    else:
                        lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
                    print(f"Learning rate: {lr}")
                period_loss = 0
                print(f"Memory allocated: {torch.cuda.memory_allocated()//(1024**2)}", flush=True)  # 对于不用LOMO的optimizer，这里第二次和第一次输出的差就是optimizer states占的空间，用LOMO则states在第一次输出前的backward过程中就存在了
                print(f"Memory reserved: {torch.cuda.memory_reserved()//(1024**2)}", flush=True)

            # The below code is only executed during the update step
            # add grad norm clipping (a kind of grad normalization to make grad norm less or equal to the clipping limit) 我们先不用
            if args.grad_clipping != 0.0: torch.nn.utils.clip_grad_norm_(trainable_params_float, args.grad_clipping)

            if global_rank == 0: pbar.update(1)
            
            if not layer_wise_flag:
                # optimizer.step()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()   # 注意：q-adam-mini的zero_grad()默认将grad设置为None，即会清除其内存占用。不知为何，设置为0还是None对allocated和reserved显存都没有影响

            update_step += 1
            update_time = time.time() - update_time

            # save checkpoint by save_every in the main training process
            if local_step > args.gradient_accumulation and update_step % args.save_every == 0 and global_rank == 0:
                current_model_directory = f"{args.save_dir}/model_{args.name}_{update_step}"
                logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
                os.makedirs(current_model_directory, exist_ok=True)
                # model.save_pretrained(current_model_directory, max_shard_size='100GB')
                saving_model_weight(model, f"{current_model_directory}/pytorch_model.bin")

                if not layer_wise_flag:
                    optimizer_checkpoint = {
                        "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict(),
                        "update_step": update_step,
                        "global_step": global_step,
                        "config": run_config,
                        "wandb": wandb.run.dir if not args.unset_wandb else None,
                        "dtype": args.dtype,
                    }
                    torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)
                    
                # save wandb related info
                if not args.unset_wandb:
                    wandb_info = {
                        "wandb_id": wandb.run.id,
                    }
                    with open(f"{current_model_directory}/wandb.json", "w") as f:
                        json.dump(wandb_info, f, indent=4)

            # evaluation
            if update_step % args.eval_every == 0:
                logger.info(f"Performing evaluation at step {update_step}")
                total_loss, evaluated_on_tokens = evaluate_model(
                    model, tokenizer, pad_idx, global_rank, world_size, device, args.batch_size
                )
                if global_rank == 0:
                    if not args.unset_wandb:
                        wandb.log({
                            "final_eval_loss": total_loss,
                            "final_eval_tokens": evaluated_on_tokens,
                            },
                            step=global_step,
                        )
                logger.info(f"Eval loss at step {update_step}: {total_loss}")

            if not layer_wise_flag:
                lr = optimizer.param_groups[0]["lr"]
            else:
                lr = list(optimizer_dict.values())[0].param_groups[0]["lr"]
            tokens_in_update = tokens_seen - tokens_seen_before
            tokens_seen_before = tokens_seen
            batches_in_update = args.gradient_accumulation * world_size

            if global_rank == 0:
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

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")
    if global_rank == 0: pbar.close()
    
    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    
    total_loss, evaluated_on_tokens = evaluate_model(
        model, tokenizer, pad_idx, global_rank, world_size, device, args.batch_size
    )

    if global_rank == 0:
        if not args.unset_wandb:
            wandb.log({
                "final_eval_loss": total_loss,
                "final_eval_tokens": evaluated_on_tokens,
                },
                step=global_step,
            )
        logger.info(f"Final eval loss: {total_loss}")

    # 主训练进程完成模型、优化器等的checkpoint数据和信息的保存
    if global_rank == 0:
        current_model_directory = f"{args.save_dir}/model_{args.name}_{update_step}"
        logger.info(f"Saving model and optimizer to {current_model_directory}, update step {update_step}")
        os.makedirs(current_model_directory, exist_ok=True)
        # model.save_pretrained(current_model_directory)  # huggingface的save_pretrained在存模型的时候也是存的state_dict()的返回结果
        saving_model_weight(model, f"{current_model_directory}/pytorch_model.bin")
        
        if not layer_wise_flag:
            optimizer_checkpoint = {
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
                "update_step": update_step,
                "global_step": global_step,
                "config": run_config,
                "wandb": wandb.run.dir if not args.unset_wandb else None,
                "dtype": args.dtype,
            }
            torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # del loss, optimizer, scheduler
    # import gc; gc.collect() # 手动触发python内存垃圾回收机制
    torch.cuda.empty_cache()

    logger.info("Script finished successfully")
    print(f"Rank {global_rank} finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)