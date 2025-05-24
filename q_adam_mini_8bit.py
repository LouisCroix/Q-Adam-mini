import math
from typing import Iterable, Tuple, Union, Optional
from collections import defaultdict
import os
import wandb

from bitsandbytes.optim.optimizer import Optimizer2State
import bitsandbytes.functional as F

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed._tensor import Replicate
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class QAdamMini8bit(Optimizer2State):
    
    def __init__(
            self,
            named_parameters: Iterable[Tuple[str, nn.Parameter]],
            lr: Union[float, torch.Tensor] = 1e-3,
            betas: Tuple[float, float] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 1e-2,
            *,
            model_sharding: bool = None,
            dim: int = 2048,
            n_heads: int = 32,
            n_kv_heads: Optional[int] = None,
            stochastic_round_state: bool = True,
            otherwise_dtype=torch.bfloat16,
            default_group_size=256,
            verbose=True,
            optim_bits=8, args=None, min_8bit_size=4096, percentile_clipping=100, block_wise=True, is_paged=False
    ):

        '''
        This is the official implementation of Adam-mini (version 1.1.0).

        Paper: [Adam-mini: Use Fewer Learning Rates To Gain More](https://arxiv.org/abs/2406.16793).

        Github repo: https://github.com/zyushun/Adam-mini

        Arguments:
            named_parameters ('Iterable[Tuple[str, nn.Parameter]]'): Iterable of named parameters to optimize or dictionaries defining parameter groups. Usually set to model.named_parameters()

            lr (`float`, *optional*, defaults to 0.001): The learning rate to use.

            betas (`Tuple[float,float]`, *optional*, defaults to `(0.9, 0.999)`): Same as Adam's betas parameters (b1, b2).

            eps (`float`, *optional*, defaults to 1e-06): Same as Adam's epsilon for numerical stability.

            weight_decay (`float`, *optional*, defaults to 0.0): Decoupled weight decay to apply.

            model_sharding (`bool`, *optional*, defaults to None): Set to True if you are using model parallelism with more than 1 GPU, including FSDP and zero_1,2,3 in Deepspeed. Set to False if otherwise. Due to the historical reason, this argument is deprecated since version 1.0.2. We will assume that model parallelism is always used. We will remove this argument in the future version.

            dim (`int`, *optional*, defaults to 2048): Dimension for hidden features. Can be left unspecified if training non-transformer models.

            n_heads (`int`, *optional*, defaults to 32): Number of attention heads. Can be left unspecified if training non-transformer models.

            n_kv_heads (`int`, *optional*, defaults to None): Number of heads for Key and Value. Or equivalently, number of query groups in Group Query Attention. Also known as "n_query_groups". If not specified, it will be equal to n_head. Can be left unspecified if training non-transformer models.
            Group Query Attention: 一种节约显存的注意机制实现方法，只提供n_kv_heads个不同的key向量和value向量，一对key和value向量由 n_heads/n_kv_heads 个 query向量共享。

        Example:

        ```python
        optimizer = Adam_mini(
                    named_parameters = model.named_parameters(),
                    lr = lr,
                    betas = (beta1,beta2),
                    eps = eps,
                    weight_decay = weight_decay,
                    dim = model_config.dim,
                    n_heads = model_config.n_heads,
                    n_kv_heads = model_config.n_kv_heads,
                    )
        ```

        '''

        self.dim = dim
        self.n_heads = n_heads
        if n_kv_heads is not None:
            assert n_heads % n_kv_heads == 0, f"{n_heads} {n_kv_heads}"
            self.n_kv_heads = n_kv_heads
        else:
            self.n_kv_heads = n_heads
        self.stochastic_round_state = stochastic_round_state
        self.default_group_size = default_group_size
        self.min_8bit_size = min_8bit_size
        self.otherwise_dtype = otherwise_dtype
        print(f"The otherwise_dtype is {self.otherwise_dtype}, which is used for all tensors not quantized in this optimizer.")

        self.world_size = torch.cuda.device_count()
        self.verbose = verbose

        assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
        self.head_numel = self.dim * self.dim // self.n_heads
        
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not self.dim == int(self.dim):
            raise ValueError("Invalid dim value: {}".format(self.dim))
        if not self.n_heads == int(self.n_heads):
            raise ValueError("Invalid n_heads value: {}".format(self.n_heads))
        if not self.n_kv_heads == int(self.n_kv_heads):
            raise ValueError("Invalid n_kv_heads value: {}".format(self.n_kv_heads))

        if model_sharding is not None and verbose:
            print(
                "UserWarning: model_sharding is deprecated since version 1.0.2. This argument is always set True. We will remove this argument in the future version.")

        # Embedding layer. Use one lr per token
        self.embd_names = {"embed", "embd", "wte"}  # move to mlp
        # Output layers. Use one lr per token
        self.output_names = {"lm_head.weight", "output.weight"}  # move output to mlp
        # Query and Keys. User one lr per head
        self.wqk_names = {"k_proj.weight", "q_proj.weight", "wq.weight", "wk.weight"}
        # Values. Use one lr per neuron
        # it is okay to set self.wv_names to be empty and use a single lr for the whole v. But this  will bring extra all_reduce operations
        # self.wv_names = {"v_proj.weight", "wv.weight"}
        self.wv_names = {}
        # attn_proj. Use one lr per neuron
        self.attn_proj_names = {"o_proj.weight", "wo.weight", "attn.proj.weight"}
        # MLPs. Use one lr per neuron
        self.mlp_names = {"feed_forward", "linear", "mlp"}
        # Blocks that use Adam. For old versions before v.1.1.0, this is for embedding layer and output layer. For the current version, this is empty
        self.adam_block_names = {}

        optim_groups = []
        count_embd = count_output = count_wqk = count_wv = count_attn_proj = count_mlp = 0
        for param_name, param in named_parameters:
            if (not hasattr(param, 'group_size')) and not param.requires_grad:
                continue
            if verbose:
                print('Adam-mini found the param block with name:', param_name, param.size())
            state = {}
            state["name"] = param_name
            state["params"] = param
            state["beta1"] = betas[0]
            state["beta2"] = betas[1]
            state["lr"] = lr
            state["eps"] = eps
            if "norm" in param_name or "ln_f" in param_name:
                state["weight_decay"] = 0.0
            else:
                state["weight_decay"] = weight_decay
                
            if any(embd_name in param_name for embd_name in self.embd_names):
                count_embd += 1
            if any(output_name in param_name for output_name in self.output_names):
                count_output += 1
            if any(wqk_name in param_name for wqk_name in self.wqk_names):
                count_wqk += 1
                assert (self.dim * self.dim) % self.n_heads == 0, f"{self.dim} {self.n_heads}"
            if any(wv_name in param_name for wv_name in self.wv_names):
                count_wv += 1
            if any(attn_proj_name in param_name for attn_proj_name in self.attn_proj_names):
                count_attn_proj += 1
            if any(mlp_name in param_name for mlp_name in self.mlp_names):
                count_mlp += 1
            #     state["neuron_numel"] = self.dim

            optim_groups.append(state)

        if self.verbose:
            print(
                f'Adam-mini found {count_embd} embedding layers, {count_output} output layers; {count_wqk} Querys and Keys;  {count_wv} Values;  {count_attn_proj} attn_proj;  {count_mlp} MLPs;')

        if count_embd == 0 and self.verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No embedding layer found. If you are training Transformers, please check the name of your embedding layer and manually add them to 'self.embd_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.embd_names.add('the keywords in the name of your embedding layer'). ")
        if count_output == 0 and self.verbose:
            # warning
            print(
                "=====>>> Warning by Adam-mini: No output layer found. If you are training Transformers (without weight-tying), please check the name of your output layer and manually add them to 'self.output_names' of Adam-mini. You can do this by adding an additional line of code: optimizer.output_names.add('the keywords in the  name of your output layer').  Please ignore this warning if you are using weight-tying.")
        if count_wqk == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Query or Key found. If you are training Transformers, please check the name of your Query and Key in attention blocks and manually add them to 'self.wqk_names' of Adam-mini. You can do this by adding two additional lines of code: optimizer.wqk_names.add('the keywords in the  name of your Query' ); optimizer.wqk_names.add('the keywords in the  name of your Key'). ")

        if count_wv == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No Value found. If you are training Transformers, please check the name of your Value in attention blocks and manually add them to 'self.wv_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.wv_names.add('the keywords in the  name of your Value' ). ")

        if count_attn_proj == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No attn_proj found. If you are training Transformers, please check the name of your attn_proj in attention blocks and manually add them to 'self.attn_proj_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.attn_proj_names.add('the keywords in the  name of your attn_proj' ). ")

        if count_mlp == 0 and self.verbose:
            # warning
            print(
                "=====>>>  Warning by Adam-mini: No MLP found. If you are training Transformers, please check the name of your MLP in attention blocks and manually add them to 'self.mlp_names' of Adam-mini. You can do this by adding an additional lines of code: optimizer.attn_proj_names.add('the keywords in the  name of your MLP' ). ")

        if (count_output + count_embd + count_wqk + count_wv + count_attn_proj + count_mlp == 0) and self.verbose:
            print(
                "=====>>>  Warning by Adam-mini: you are using default PyTorch partition for Adam-mini. It can cause training instability on large-scale Transformers.")

        assert optim_bits == 8 or optim_bits == 4, f"optim_bits can only be 8 or 4, found {optim_bits}"
        self.dtype = torch.uint8 if optim_bits == 8 else torch.uint4
        super().__init__("adam", optim_groups, lr, betas, eps, weight_decay, 8, args, min_8bit_size, percentile_clipping, block_wise, is_paged=is_paged)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
                
        if not self.initialized:
            self.check_overrides()
            self.to_gpu()  # needed for fairseq pure fp16 training
            self.initialized = True
                
        for gindex, group in enumerate(self.param_groups):
            beta1 = group["beta1"]
            beta2 = group["beta2"]
            lr = group["lr"]
            name = group["name"]
            eps = group["eps"]

            for pindex, p in enumerate(group["params"]):
                
                flag_use_float_grad = hasattr(p, "float_grad")
                use_quant_state = (p.numel() >= self.min_8bit_size)
                using_dtype = self.dtype if use_quant_state else self.otherwise_dtype
                
                if flag_use_float_grad:
                    grad_dtype = p.float_grad.dtype
                    p.float_grad = p.float_grad.to(dtype=self.otherwise_dtype)
                    
                    # change p.data to float weight
                    try:
                        num_ranks = dist.get_world_size()
                    except:
                        num_ranks = 1

                    if num_ranks > 1:
                        grad_list = [torch.zeros_like(p.float_grad) for _ in range(num_ranks)]
                        dist.all_gather(grad_list, p.float_grad)
                        p.float_grad.data.copy_(sum(grad_list)/num_ranks)

                    float_weight = self._dequantize(p.data, p.float_grad.dtype, p.group_size, p.scales, p.zeros)
                    p.data = p.data.to(p.float_grad.dtype)
                    p.data = float_weight.clone().to(p.data.device)
                #     print("float_grad:", p.float_grad.dtype)
                # else:
                #     print("grad:", p.grad.dtype)
                    
                state = self.state[p]
                assert "vmean" not in state or state["vmean"].dtype == self.otherwise_dtype, state["vmean"].dtype
                # if "step" not in state:
                #     state["step"] = 0
                # if 'state1' not in state:
                #     self.init_state(group, p, gindex, pindex)
                    
                self.prefetch_state(p)

                if any(adam_block_name in name for adam_block_name in
                       self.adam_block_names):  # For v.1.1.0, we will not enter here
                    if (not flag_use_float_grad) and p.grad is None:
                        continue
                    if len(state) == 0:
                        self.init_state(group, p, gindex, pindex)
                        state["step"] = 0
                    
                    self.update_step(group, p, gindex, pindex, flag_use_float_grad=flag_use_float_grad)

                elif any(wqk_name in name for wqk_name in self.wqk_names):  # this is for query and key
                    if (not flag_use_float_grad) and p.grad is None:
                        continue
                    head_numel = self.head_numel
                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, dtype=using_dtype, memory_format=torch.preserve_format).view(-1, head_numel)
                        state["head_per_gpu"] = state["m"].size(0)  # this is head per gpu
                        state["step"] = 0
                        state["vmean"] = torch.zeros_like(state["m"][0:state["head_per_gpu"], 0:1], dtype=self.otherwise_dtype,
                                                          memory_format=torch.preserve_format)
                        
                        if use_quant_state:
                            if flag_use_float_grad:
                                state["m_scales"] = torch.zeros_like(p.scales, memory_format=torch.preserve_format)
                                state["m_zeros"] = torch.zeros_like(p.zeros, memory_format=torch.preserve_format)
                                state["m_group_size"] = p.group_size
                            else:
                                state["m"], state["m_scales"], state["m_zeros"] = self._quantize(state["m"], q_group_size=self.default_group_size)
                                state["m_group_size"] = self.default_group_size

                    if flag_use_float_grad:
                        grad = p.float_grad # float
                    else:
                        grad = p.grad       # float
                    head_per_gpu = state["head_per_gpu"]
                    grad = grad.view(head_per_gpu, head_numel)
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)
                    
                    if use_quant_state:
                        float_m = self._dequantize(state["m"].data, grad.dtype, state["m_group_size"], state["m_scales"], state["m_zeros"])
                    else:
                        float_m = state["m"].data

                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    float_m.lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(head_per_gpu, 1)
                    update = (float_m * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)
                    
                    if use_quant_state:
                        if self.stochastic_round_state:
                            state["m"].data, state["m_scales"], state["m_zeros"] = self._quantize_stochastic_round(float_m, q_group_size=state["m_group_size"])
                        else:
                            state["m"].data, state["m_scales"], state["m_zeros"] = self._quantize(float_m, q_group_size=state["m_group_size"])
                    else:
                        state["m"].data = float_m

                elif any(embd_name in name for embd_name in self.embd_names) or any(
                        output_name in name for output_name in self.output_names) or any(
                        wv_name in name for wv_name in self.wv_names) or any(
                        mlp_name in name for mlp_name in self.mlp_names) or any(
                        attn_proj_name in name for attn_proj_name in self.attn_proj_names):  # MLP blocks. If True, then Adam-mini will use one lr per output neuron. This will avoid all_reduce when the single MLP block is sharded on different GPUs. However, we find this design will not boost performance. By default, we will not enter here
                    if (not flag_use_float_grad) and p.grad is None:
                        continue

                    if len(state) == 0:
                        state["m"] = torch.zeros_like(p, dtype=using_dtype, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["neuron_per_gpu"] = state["m"].size(0)  # this is neuron per gpu
                        state["vmean"] = torch.zeros_like(state["m"][0:state["neuron_per_gpu"], 0:1], dtype=self.otherwise_dtype,
                                                          memory_format=torch.preserve_format)
                        
                        if use_quant_state:
                            if flag_use_float_grad:
                                state["m_scales"] = torch.zeros_like(p.scales, memory_format=torch.preserve_format)
                                state["m_zeros"] = torch.zeros_like(p.zeros, memory_format=torch.preserve_format)
                                state["m_group_size"] = p.group_size
                            else:
                                state["m"], state["m_scales"], state["m_zeros"] = self._quantize(state["m"], q_group_size=self.default_group_size)
                                state["m_group_size"] = self.default_group_size

                    if flag_use_float_grad:
                        grad = p.float_grad # float
                    else:
                        grad = p.grad       # float
                    neuron_per_gpu = state["neuron_per_gpu"]
                    tmp_lr = torch.mean(grad * grad, dim=1, keepdim=True)
                    
                    if use_quant_state:
                        float_m = self._dequantize(state["m"].data, grad.dtype, state["m_group_size"], state["m_scales"], state["m_zeros"])
                    else:
                        float_m = state["m"].data
                    
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    state["step"] += 1
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    float_m.lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = ((1 / bias_correction_1) / h).view(neuron_per_gpu, 1)
                    update = (float_m * stepsize).view(p.size())
                    update.mul_(lr)
                    p.add_(-update)
                    
                    if use_quant_state:
                        if self.stochastic_round_state:
                            state["m"].data, state["m_scales"], state["m_zeros"] = self._quantize_stochastic_round(float_m, q_group_size=state["m_group_size"])
                        else:
                            state["m"].data, state["m_scales"], state["m_zeros"] = self._quantize(float_m, q_group_size=state["m_group_size"])
                    else:
                        state["m"].data = float_m

                else:  # # other blocks. By default, this is for LayerNorms. Sometimes it is also fine to put Value here.
                    if len(state) == 0:
                        block_numel = torch.tensor(p.numel()).to(torch.float32).to(device)
                        reduced = False
                        if (self.world_size > 1):
                            tensor_list = [torch.zeros_like(block_numel) for _ in range(self.world_size)]

                            dist.all_gather(tensor_list, block_numel)
                            s = 0
                            block_numel = 0
                            for d in tensor_list:
                                if (d > 0):
                                    s = s + 1
                                block_numel = block_numel + d
                            if (s >= 2):
                                reduced = True

                        state["m"] = torch.zeros_like(p, dtype=using_dtype, memory_format=torch.preserve_format)
                        state["step"] = 0
                        state["reduced"] = reduced

                        state["vmean"] = torch.zeros_like(torch.sum(p * p), dtype=self.otherwise_dtype, memory_format=torch.preserve_format)
                        state["block_numel"] = block_numel.item()
                        
                        if use_quant_state:
                            if flag_use_float_grad:
                                state["m_scales"] = torch.zeros_like(p.scales, memory_format=torch.preserve_format)
                                state["m_zeros"] = torch.zeros_like(p.zeros, memory_format=torch.preserve_format)
                                state["m_group_size"] = p.group_size
                            else:
                                state["m"], state["m_scales"], state["m_zeros"] = self._quantize(state["m"], q_group_size=self.default_group_size)
                                state["m_group_size"] = self.default_group_size
                        
                    if (not flag_use_float_grad) and p.grad is None:
                        tmp_lr = torch.zeros_like(torch.sum(state["vmean"] * state["vmean"]), dtype=self.otherwise_dtype)
                    else:
                        if flag_use_float_grad:
                            grad = p.float_grad # float
                        else:
                            grad = p.grad       # float
                        tmp_lr = torch.sum(grad * grad)

                    if (state["reduced"]):
                        # Force communication over GPUs when GPUs are available
                        if tmp_lr.device.type == 'cpu':
                            # Move the tensor to the current GPU device
                            tmp_lr_gpu = tmp_lr.to(torch.cuda.current_device())

                            if "device_mesh" in dir(tmp_lr):
                                # when tmp_lr is a  DTensor in TorchTitan
                                lr_local = tmp_lr.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr.redistribute(placements=[Replicate()])
                            else:
                                # when tmp_lr is a  standard tensor
                                dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                            # Move the result back to the CPU tensor
                            tmp_lr.copy_(tmp_lr_gpu.cpu())
                        else:
                            # Tensor is already on GPU, use NCCL backend
                            if "device_mesh" in dir(tmp_lr):
                                # when tmp_lr is a  DTensor in TorchTitan
                                lr_local = tmp_lr.to_local()
                                dist.all_reduce(lr_local, op=dist.ReduceOp.SUM)
                                tmp_lr.redistribute(placements=[Replicate()])
                            else:
                                # when tmp_lr is a  standard tensor
                                dist.all_reduce(tmp_lr, op=dist.ReduceOp.SUM)

                    if (not flag_use_float_grad) and p.grad is None:
                        continue
                    tmp_lr = tmp_lr / state["block_numel"]

                    if use_quant_state:
                        float_m = self._dequantize(state["m"].data, grad.dtype, state["m_group_size"], state["m_scales"], state["m_zeros"])
                    else:
                        float_m = state["m"].data
                        
                    if group["weight_decay"] > 0.0:
                        p.mul_(1 - lr * group["weight_decay"])
                    state["step"] += 1

                    float_m.lerp_(grad, 1 - beta1)
                    bias_correction_1 = 1 - beta1 ** state["step"]
                    bias_correction_2 = 1 - beta2 ** state["step"]
                    bias_correction_2_sqrt = math.sqrt(bias_correction_2)
                    state["vmean"].mul_(beta2).add_(tmp_lr, alpha=1 - beta2)
                    h = (state["vmean"].sqrt() / bias_correction_2_sqrt).add_(eps)
                    stepsize = (1 / bias_correction_1) / h
                    update = float_m * (stepsize.to(float_m.device))
                    update.mul_(lr)
                    p.add_(-update)
                    
                    if use_quant_state:
                        if self.stochastic_round_state:
                            state["m"].data, state["m_scales"], state["m_zeros"] = self._quantize_stochastic_round(float_m, q_group_size=state["m_group_size"])
                        else:
                            state["m"].data, state["m_scales"], state["m_zeros"] = self._quantize(float_m, q_group_size=state["m_group_size"])
                    else:
                        state["m"].data = float_m
                        
                torch.cuda.synchronize()

                if flag_use_float_grad:
                    # quantize gradient-updated p.data back to int8
                    saved_data = p.data.clone()
                    if p.stochastic_round:
                        p.data, p.scales, p.zeros = self._quantize_stochastic_round(saved_data, q_group_size=p.group_size)
                    else:
                        p.data, p.scales, p.zeros = self._quantize(saved_data, q_group_size=p.group_size)
                    p.float_grad = p.float_grad.to(dtype=grad_dtype)
                    
        if self.is_paged:
            # all paged operation are asynchronous, we need
            # to sync to make sure all tensors are in the right state
            torch.cuda.synchronize()
                    
        return loss
    
    @torch.no_grad()
    def update_step(self, group, p, gindex, pindex, flag_use_float_grad=False):
        state = self.state[p]

        if flag_use_float_grad:
            grad = p.float_grad
        else:
            grad = p.grad

        config = self.get_config(gindex, pindex, group)

        state["step"] += 1
        step = state["step"]

        if config["percentile_clipping"] < 100:
            current_gnorm, clip_value, gnorm_scale = F.percentile_clipping(
                grad,
                state["gnorm_vec"],
                step,
                config["percentile_clipping"],
            )
        else:
            gnorm_scale = 1.0

        if state["state1"].dtype == torch.float:
            F.optimizer_update_32bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                config["betas"][0],
                config["eps"],
                step,
                config["lr"],
                state["state2"],
                config["betas"][1],
                config["weight_decay"],
                gnorm_scale,
                state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
                skip_zeros=config["skip_zeros"],
            )

        elif state["state1"].dtype == torch.uint8 and not config["block_wise"]:
            F.optimizer_update_8bit(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["max1"],
                state["max2"],
                state["new_max1"],
                state["new_max2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                unorm_vec=state["unorm_vec"] if config["max_unorm"] > 0.0 else None,
                max_unorm=config["max_unorm"],
            )

            # swap maxes
            state["max1"], state["new_max1"] = state["new_max1"], state["max1"]
            state["max2"], state["new_max2"] = state["new_max2"], state["max2"]
        elif state["state1"].dtype == torch.uint8 and config["block_wise"]:
            F.optimizer_update_8bit_blockwise(
                self.optimizer_name,
                grad,
                p,
                state["state1"],
                state["state2"],
                config["betas"][0],
                config["betas"][1],
                config["eps"],
                step,
                config["lr"],
                state["qmap1"],
                state["qmap2"],
                state["absmax1"],
                state["absmax2"],
                config["weight_decay"],
                gnorm_scale=gnorm_scale,
                skip_zeros=config["skip_zeros"],
            )
            
    @torch._disable_dynamo
    def zero_grad(self, set_to_none: bool = True) -> None:
        r"""Resets the gradients of all optimized :class:`torch.Tensor` s.

        Args:
            set_to_none (bool): instead of setting to zero, set the grads to None.
                This will in general have lower memory footprint, and can modestly improve performance.
                However, it changes certain behaviors. For example:
                1. When the user tries to access a gradient and perform manual ops on it,
                a None attribute or a Tensor full of 0s will behave differently.
                2. If the user requests ``zero_grad(set_to_none=True)`` followed by a backward pass, ``.grad``\ s
                are guaranteed to be None for params that did not receive a gradient.
                3. ``torch.optim`` optimizers have a different behavior if the gradient is 0 or None
                (in one case it does the step with a gradient of 0 and in the other it skips
                the step altogether).
        """
        
        foreach = self.defaults.get("foreach", False) or self.defaults.get(
            "fused", False
        )

        if not hasattr(self, "_zero_grad_profile_name"):
            self._patch_step_function()

        per_device_and_dtype_grads: Optional[
            DefaultDict[torch.device, DefaultDict[torch.dtype, List[torch.Tensor]]]
        ]
        if foreach:
            per_device_and_dtype_grads = defaultdict(lambda: defaultdict(list))
        else:
            per_device_and_dtype_grads = None

        with torch.autograd.profiler.record_function(self._zero_grad_profile_name):
            for group in self.param_groups:
                for p in group["params"]:
                    flag_use_float_grad = hasattr(p, "float_grad")
                    if flag_use_float_grad:
                        if p.float_grad is not None:
                            if set_to_none:
                                p.float_grad = None
                    else:
                        if p.grad is not None:
                            if set_to_none:
                                p.grad = None
                            else:
                                if p.grad.grad_fn is not None:
                                    p.grad.detach_()
                                else:
                                    p.grad.requires_grad_(False)
                                if not foreach or p.grad.is_sparse:
                                    p.grad.zero_()
                                else:
                                    assert per_device_and_dtype_grads is not None
                                    per_device_and_dtype_grads[p.grad.device][
                                        p.grad.dtype
                                    ].append(p.grad)
            if foreach:
                assert per_device_and_dtype_grads is not None
                for per_dtype_grads in per_device_and_dtype_grads.values():
                    for grads in per_dtype_grads.values():
                        torch._foreach_zero_(grads)

    @torch.no_grad()
    def _quantize(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)

        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0, f"{scales}"
        assert torch.isnan(w).sum() == 0, f"{w}"

        w = torch.clamp(torch.round(w / scales) + zeros, min_int, max_int)
        w = w.reshape(org_w_shape).to(torch.uint8)

        return w, scales, zeros

    @torch.no_grad()
    def _quantize_stochastic_round(self, w, q_group_size=-1, n_bit=8):
        org_w_shape = w.shape
        if q_group_size > 0:
            assert w.nelement() % q_group_size == 0
            w = w.reshape(-1, q_group_size)
        assert w.dim() == 2

        max_val = w.amax(dim=1, keepdim=True)
        min_val = w.amin(dim=1, keepdim=True)
        max_int = 2**n_bit - 1
        min_int = 0
        scales = (max_val - min_val).clamp(min=1e-5) / max_int
        zeros = (-torch.round(min_val / scales)).clamp_(min_int, max_int)

        assert torch.isnan(scales).sum() == 0
        assert torch.isnan(w).sum() == 0

        # Stochastic Rounding
        w_round = w / scales
        up_round_w = torch.ceil(w_round)
        down_round_w = torch.floor(w_round)
        probability = (w_round - down_round_w)
        random = torch.rand_like(probability)
        w = torch.where(random < probability, up_round_w, down_round_w)

        w = torch.clamp(w + zeros, min_int, max_int)
        w = w.reshape(org_w_shape).to(torch.uint8)

        return w, scales, zeros

    @torch.no_grad()
    def _dequantize_and_update(self, weight, weight_update, group_size, scales, zeros):
        float_weight = weight.to(weight_update.dtype).reshape(-1, group_size)   
        (float_weight.sub_(zeros)).mul_(scales)
        float_weight = float_weight.reshape(weight.shape)
        return float_weight + weight_update

    @torch.no_grad()
    def _dequantize(self, weight, dtype, group_size, scales, zeros):
        float_weight = weight.to(dtype).reshape(-1, group_size)
        (float_weight.sub_(zeros)).mul_(scales)
        float_weight = float_weight.reshape(weight.shape)
        return float_weight
