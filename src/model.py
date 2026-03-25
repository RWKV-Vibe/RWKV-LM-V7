#################################################################
# The RWKV Language Model - https://github.com/BlinkDL/RWKV-LM
#################################################################

import gc
import importlib
import math
import os

import pytorch_lightning as pl
import torch
import torch.nn as nn
from pytorch_lightning.strategies import DeepSpeedStrategy
from rwkvfla.modules.token_shift import token_shift
from rwkvfla.ops.rwkv7.fused_addcmul import fused_addcmul_rwkv7
from rwkvfla.ops.rwkv7.fused_k_update import fused_k_rwkv7
from torch.nn import functional as F
from torch.utils.cpp_extension import load

if importlib.util.find_spec("deepspeed"):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

try:
    print("RWKV_MY_TESTING", os.environ["RWKV_MY_TESTING"])
except BaseException:
    os.environ["RWKV_MY_TESTING"] = ""

def __nop(ob):
    return ob


ROCm_flag = torch.version.hip is not None
CompileFunction = __nop
if os.environ["RWKV_COMPILE_ON"] == "1":
    CompileFunction = torch.compile
    import torch._dynamo.config
    torch._dynamo.config.allow_unspec_int_on_nn_module = True


#################################################################
# CUDA Kernel
#################################################################


HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE"])

if "x070" in os.environ["RWKV_MY_TESTING"]:
    CHUNK_LEN = 16

    if ROCm_flag is True:
        flags = [
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "-xhip",
            "-fopenmp",
            "-ffast-math",
            "-O3",
            "-munsafe-fp-atomics",
            "--save-temps",
            '-DAMD'
        ]
        load(
            name="wind_backstepping",
            sources=["cuda/wkv7_hip.hip", "cuda/wkv7_op.hip"],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )
    else:
        flags = [
            "-res-usage",
            f"-D_C_={HEAD_SIZE}",
            f"-D_CHUNK_LEN_={CHUNK_LEN}",
            "--use_fast_math",
            "-O3",
            "-Xptxas -O3",
            "--extra-device-vectorization",
        ]
        load(
            name="wind_backstepping",
            sources=["cuda/wkv7_cuda.cu", "cuda/wkv7_op.cpp"],
            is_python_module=False,
            verbose=True,
            extra_cuda_cflags=flags,
        )

    class WindBackstepping(torch.autograd.Function):
        @staticmethod
        def forward(ctx, w, q, k, v, z, b):
            B, T, H, C = w.shape
            assert T % CHUNK_LEN == 0
            assert all(i.dtype == torch.bfloat16 for i in [w, q, k, v, z, b])
            assert all(i.is_contiguous() for i in [w, q, k, v, z, b])
            y = torch.empty_like(v)
            s = torch.empty(
                B, H, T // CHUNK_LEN, C, C, dtype=torch.float32, device=w.device
            )
            sa = torch.empty(B, T, H, C, dtype=torch.float32, device=w.device)
            torch.ops.wind_backstepping.forward(w, q, k, v, z, b, y, s, sa)
            ctx.save_for_backward(w, q, k, v, z, b, s, sa)
            return y

        @staticmethod
        def backward(ctx, dy):
            assert all(i.dtype == torch.bfloat16 for i in [dy])
            assert all(i.is_contiguous() for i in [dy])
            w, q, k, v, z, b, s, sa = ctx.saved_tensors
            dw, dq, dk, dv, dz, db = [torch.empty_like(x) for x in [
                w, q, k, v, z, b]]
            torch.ops.wind_backstepping.backward(
                w, q, k, v, z, b, dy, s, sa, dw, dq, dk, dv, dz, db
            )
            return dw, dq, dk, dv, dz, db

    def RUN_CUDA_RWKV7g(q, w, k, v, a, b):
        B, T, HC = q.shape
        q, w, k, v, a, b = [i.view(B, T, HC // 64, 64)
                            for i in [q, w, k, v, a, b]]
        return WindBackstepping.apply(w, q, k, v, a, b).view(B, T, HC)


#################################################################


class RWKV_Tmix_x070(nn.Module):
    @torch.no_grad()
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.my_testing = args.my_testing

        self.head_size = args.head_size
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        H = self.n_head
        N = self.head_size
        C = args.n_embd

        ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
        ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, C)
        for i in range(C):
            ddd[0, 0, i] = i / C

        self.x_r = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))
        self.x_w = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_v = nn.Parameter(1.0 - torch.pow(ddd, 0.7 * ratio_1_to_almost0))
        self.x_a = nn.Parameter(1.0 - torch.pow(ddd, 0.9 * ratio_1_to_almost0))
        self.x_g = nn.Parameter(1.0 - torch.pow(ddd, 0.2 * ratio_1_to_almost0))

        def ortho_init(x, scale):
            shape = x.shape
            if len(shape) == 2:
                gain = (
                    math.sqrt(shape[0] / shape[1]
                              ) if shape[0] > shape[1] else 1
                )
                nn.init.orthogonal_(x, gain=gain * scale)
            elif len(shape) == 3:
                gain = (
                    math.sqrt(shape[1] / shape[2]
                              ) if shape[1] > shape[2] else 1
                )
                for i in range(shape[0]):
                    nn.init.orthogonal_(x[i], gain=gain * scale)
            else:
                assert False
            return x

        www = torch.zeros(C)
        zigzag = torch.zeros(C)
        linear = torch.zeros(C)
        for n in range(C):
            linear[n] = n / (C - 1) - 0.5
            zigzag[n] = ((n % N) - ((N - 1) / 2)) / ((N - 1) / 2)
            zigzag[n] = zigzag[n] * abs(zigzag[n])
            www[n] = -6 + 6 * (n / (C - 1)) ** (1 + 1 * ratio_0_to_1**0.3)

        # Increase lora dimension for headdim>64
        factor = self.head_size / 64
        D_DECAY_LORA = max(
            32, int(round((2.5 * (C**0.5)) * factor / 32) * 32))  # suggestion
        self.w1 = nn.Parameter(torch.zeros(C, D_DECAY_LORA))
        self.w2 = nn.Parameter(ortho_init(torch.zeros(D_DECAY_LORA, C), 0.1))
        # !!! 0.5 comes from F.softplus !!!
        self.w0 = nn.Parameter(www.reshape(1, 1, C) + 0.5 + zigzag * 2.5)

        D_AAA_LORA = max(
            32, int(round((2.5 * (C**0.5)) * factor / 32) * 32))  # suggestion
        self.a1 = nn.Parameter(torch.zeros(C, D_AAA_LORA))
        self.a2 = nn.Parameter(ortho_init(torch.zeros(D_AAA_LORA, C), 0.1))
        self.a0 = nn.Parameter(
            torch.zeros(1, 1, C) - 0.19 + zigzag * 0.3 + linear * 0.4
        )

        D_MV_LORA = max(
            32, int(round((1.7 * (C**0.5)) * factor / 32) * 32))  # suggestion
        self.v1 = nn.Parameter(torch.zeros(C, D_MV_LORA))
        self.v2 = nn.Parameter(ortho_init(torch.zeros(D_MV_LORA, C), 0.1))
        self.v0 = nn.Parameter(torch.zeros(1, 1, C) + 0.73 - linear * 0.4)

        # Note: for some data, you can reduce D_GATE_LORA or even remove this gate
        D_GATE_LORA = max(
            32, int(round((5 * (C**0.5)) / 32) * 32))  # suggestion
        self.g1 = nn.Parameter(torch.zeros(C, D_GATE_LORA))
        self.g2 = nn.Parameter(ortho_init(torch.zeros(D_GATE_LORA, C), 0.1))

        self.k_k = nn.Parameter(torch.zeros(1, 1, C) + 0.71 - linear * 0.1)
        self.k_a = nn.Parameter(torch.zeros(1, 1, C) + 1.02)
        self.r_k = nn.Parameter(torch.zeros(H, N) - 0.04)

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(C, C, bias=False)
        self.key = nn.Linear(C, C, bias=False)
        self.value = nn.Linear(C, C, bias=False)
        self.output = nn.Linear(C, C, bias=False)
        # !!! notice eps value !!!
        self.ln_x = nn.GroupNorm(H, C, eps=64e-5)

        self.receptance.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
        self.key.weight.data.uniform_(-0.05 / (C**0.5), 0.05 / (C**0.5))
        self.value.weight.data.uniform_(-0.5 / (C**0.5), 0.5 / (C**0.5))
        self.output.weight.data.zero_()
        del www, zigzag, linear, ddd

    @CompileFunction
    def forward(self, x, v_first):
        B, T, C = x.size()
        xx = token_shift(x)
        xr, xw, xk, xv, xa, xg = fused_addcmul_rwkv7(x, xx, self.x_r,
                                                     self.x_w, self.x_k, self.x_v,
                                                     self.x_a, self.x_g)

        r = self.receptance(xr)
        # soft-clamp to (-inf, -0.5)
        w = -F.softplus(-(self.w0 + torch.tanh(xw @ self.w1) @ self.w2)) - 0.5
        k = self.key(xk)
        v = self.value(xv)
        if self.layer_id == 0:
            v_first = v  # store the v of the first layer
        else:
            v = torch.lerp(
                v, v_first, torch.sigmoid(self.v0 + (xv @ self.v1) @ self.v2)
            )  # add value residual
        # a is "in-context learning rate"
        a = torch.sigmoid(self.a0 + (xa @ self.a1) @ self.a2)
        g = torch.sigmoid(xg @ self.g1) @ self.g2

        kk = F.normalize((k * self.k_k).view(B, T, self.n_head, -1),
                         dim=-1, p=2.0).view(B, T, C)
        k = fused_k_rwkv7(k, a, self.k_a)

        x = RUN_CUDA_RWKV7g(r, w, k, v, -kk, kk * a)
        x = self.ln_x(x.view(B * T, C)).view(B, T, C)

        x = x + (
            (r.view(B, T, self.n_head, -1) * k.view(B, T, self.n_head, -1) * self.r_k).sum(
                dim=-1, keepdim=True
            )
            * v.view(B, T, self.n_head, -1)
        ).view(B, T, C)
        x = self.output(x * g)
        return x, v_first


#################################################################


class RWKV_CMix_x070(nn.Module):
    @torch.no_grad()
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
        ddd = torch.ones(1, 1, args.n_embd)
        for i in range(args.n_embd):
            ddd[0, 0, i] = i / args.n_embd
        self.x_k = nn.Parameter(1.0 - torch.pow(ddd, ratio_1_to_almost0**4))

        self.key = nn.Linear(args.n_embd, args.n_embd * 4, bias=False)
        self.value = nn.Linear(args.n_embd * 4, args.n_embd, bias=False)

        self.key.weight.data.uniform_(
            -0.5 / (args.n_embd**0.5), 0.5 / (args.n_embd**0.5)
        )
        self.value.weight.data.zero_()

    @CompileFunction
    def forward(self, x):
        xx = token_shift(x)
        k = torch.addcmul(x, xx, self.x_k)
        k = torch.relu(self.key(k)) ** 2
        return self.value(k)


#################################################################
# The RWKV Model with our blocks
#################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)

        self.att = RWKV_Tmix_x070(args, layer_id)
        self.ffn = RWKV_CMix_x070(args, layer_id)

    @CompileFunction
    def forward(self, x, v_first):
        if self.layer_id == 0:
            x = self.ln0(x)

        x_attn, v_first = self.att(self.ln1(x), v_first)
        x = x + x_attn

        x = x + self.ffn(self.ln2(x))
        return x, v_first


class L2Wrap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, loss, y):
        ctx.save_for_backward(y)
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        y = ctx.saved_tensors[0]
        # to encourage the logits to be close to 0
        factor = 1e-4 / (y.shape[0] * y.shape[1])
        maxx, ids = torch.max(y, -1, keepdim=True)
        gy = torch.zeros_like(y)
        gy.scatter_(-1, ids, maxx * factor)
        return (grad_output, gy)


class FusedLinearCrossEntropyWithL2Warp(torch.autograd.Function):
    """
    VRAM-optimized Fused Head operator.
    
    Key Features:
    1. Forward: Calculates CrossEntropy in chunks to avoid storing full Logits [B*T, V].
    2. Backward: Recomputes Logits in chunks, decoupling VRAM usage from sequence length.
    3. Integration: Built-in support for RWKV-specific L2Warp regularization.
    """
    # Chunk size 512 strikes a balance between CUDA kernel overhead and VRAM savings
    CHUNK_SIZE = 512

    @staticmethod
    def forward(ctx, hidden_states, weight, target, l2warp_factor=1e-4, chunk_size=None):
        # hidden_states: [B_T, H], weight: [V, H], target: [B_T]
        B_T, _ = hidden_states.shape
        
        ctx.save_for_backward(hidden_states, weight, target)
        ctx.l2warp_factor = l2warp_factor
        ctx.B_T = B_T
        
        chunk_size = chunk_size or FusedLinearCrossEntropyWithL2Warp.CHUNK_SIZE
        ctx.chunk_size = chunk_size

        loss_sum = torch.zeros([], device=hidden_states.device, dtype=torch.float32)
        
        for start in range(0, B_T, chunk_size):
            end = min(start + chunk_size, B_T)
            hs_chunk = hidden_states[start:end]
            target_chunk = target[start:end]
            
            # Compute logits in FP32 for numerical stability: [chunk, H] @ [H, V] -> [chunk, V]
            logits_chunk = torch.matmul(hs_chunk, weight.t()).float()
            
            # Calculate CrossEntropy sum for this chunk
            loss_chunk = F.cross_entropy(logits_chunk, target_chunk, reduction='sum')
            loss_sum += loss_chunk
        
        # Return Mean Loss
        return loss_sum / B_T

    @staticmethod
    def backward(ctx, grad_output):
        hidden_states, weight, target = ctx.saved_tensors
        l2warp_factor = ctx.l2warp_factor
        B_T = ctx.B_T
        V = weight.shape[0]
        
        chunk_size = ctx.chunk_size

        grad_hidden = torch.empty_like(hidden_states)
        grad_weight = torch.zeros_like(weight)
        
        for start in range(0, B_T, chunk_size):
            end = min(start + chunk_size, B_T)
            hs_chunk = hidden_states[start:end]
            target_chunk = target[start:end]
            
            logits_chunk = torch.matmul(hs_chunk, weight.t()).float()
            
            softmax_chunk = logits_chunk.softmax(dim=-1)
            
            grad_logits_chunk = softmax_chunk.clone()
            # Efficient in-place scatter for target subtraction
            rows = torch.arange(grad_logits_chunk.size(0), device=grad_logits_chunk.device)
            grad_logits_chunk[rows, target_chunk] -= 1.0
            grad_logits_chunk /= B_T
            
            factor = l2warp_factor / (B_T * V)
            maxx_chunk, ids_chunk = torch.max(logits_chunk, -1, keepdim=True)
            
            l2warp_grad_chunk = torch.zeros_like(logits_chunk)
            l2warp_grad_chunk.scatter_(-1, ids_chunk, maxx_chunk * factor)
            
            total_grad_logits_chunk = (grad_logits_chunk + l2warp_grad_chunk) * grad_output
            
            total_grad_logits_chunk = total_grad_logits_chunk.to(hs_chunk.dtype)
            
            grad_hidden[start:end] = torch.matmul(total_grad_logits_chunk, weight)
            grad_weight += torch.matmul(total_grad_logits_chunk.t(), hs_chunk)

        return grad_hidden, grad_weight, None, None, None


class RWKV(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args
        if not hasattr(args, "dim_att"):
            args.dim_att = args.n_embd
        # Set a sane default when the flag is missing *or* non-positive
        if not hasattr(args, "dim_ffn") or args.dim_ffn <= 0:
            # RWKV-7 uses 4x emb size, RWKV-6 uses 3.5x emb size
            multiplier = 4 if args.my_testing == "x070" else 3.5
            args.dim_ffn = int((args.n_embd * multiplier) // 32 * 32)  # multiple of 32
        assert args.n_embd % 32 == 0
        assert args.dim_att % 32 == 0
        assert args.dim_ffn % 32 == 0
        
        # Check if we should use fused L2Warp to reduce VRAM
        # Default to False if not specified
        if not hasattr(args, "fuse_l2warp"):
            args.fuse_l2warp = 0

        self.emb = nn.Embedding(args.vocab_size, args.n_embd)

        self.blocks = nn.ModuleList([Block(args, i)
                                    for i in range(args.n_layer)])

        self.ln_out = nn.LayerNorm(args.n_embd)
        self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

    def configure_optimizers(self):
        args = self.args

        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        for n, p in self.named_parameters():
            if "att.w0" in n:
                lr_2x.add(n)
            elif (
                (len(p.squeeze().shape) >= 2)
                and (args.weight_decay > 0)
                and (".weight" in n)
            ):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))

        if self.trainer.is_global_zero:
            print("decay", lr_decay, "\n")
            print("1x", lr_1x, "\n")
            print("2x", lr_2x, "\n")

        param_dict = {n: p for n, p in self.named_parameters()}

        optim_groups = [
            {
                "params": [param_dict[n] for n in lr_1x],
                "weight_decay": 0.0,
                "my_lr_scale": 1.0,
            },
            {
                "params": [param_dict[n] for n in lr_2x],
                "weight_decay": 0.0,
                "my_lr_scale": 2.0,
            },
        ]

        if args.weight_decay > 0:
            optim_groups += [
                {
                    "params": [param_dict[n] for n in lr_decay],
                    "weight_decay": args.weight_decay,
                    "my_lr_scale": 1.0,
                }
            ]
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=True,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=True,
                amsgrad=False,
            )
        else:
            if self.deepspeed_offload:
                return DeepSpeedCPUAdam(
                    optim_groups,
                    lr=self.args.lr_init,
                    betas=self.args.betas,
                    eps=self.args.adam_eps,
                    bias_correction=True,
                    adamw_mode=False,
                    weight_decay=0,
                    amsgrad=False,
                )
            return FusedAdam(
                optim_groups,
                lr=self.args.lr_init,
                betas=self.args.betas,
                eps=self.args.adam_eps,
                bias_correction=True,
                adam_w_mode=False,
                weight_decay=0,
                amsgrad=False,
            )

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    @CompileFunction
    def forward(self, idx, return_logits=True):
        args = self.args
        B, T = idx.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        x = self.emb(idx)

        v_first = torch.empty_like(x)
        for block in self.blocks:
            if args.grad_cp == 1:
                x, v_first = deepspeed.checkpointing.checkpoint(
                    block, x, v_first)
            else:
                x, v_first = block(x, v_first)

        x = self.ln_out(x)
        
        # When fuse_l2warp is True and return_logits is False,
        # we return hidden states instead of logits to save VRAM
        # The loss computation will be done in training_step with fused loss
        if args.fuse_l2warp and not return_logits:
            return x  # Return hidden states [B, T, n_embd]
        
        x = self.head(x)
        return x

    def training_step(self, batch, batch_idx):
        idx, targets = batch
        
        if hasattr(self.args, 'fuse_l2warp') and self.args.fuse_l2warp:
            # Use fused linear cross entropy with L2Warp
            # This avoids materializing the full logits tensor
            hidden = self(idx, return_logits=False)  # [B, T, n_embd]
            hidden = hidden.view(-1, hidden.shape[-1])  # 直接使用 shape[-1] 替代解包
            targets = targets.view(-1)  # [B*T]

            dynamic_l2warp = getattr(self.args, 'l2warp_factor', 1e-4)

            # Use the head weight directly to avoid storing logits
            loss = FusedLinearCrossEntropyWithL2Warp.apply(
                hidden, 
                self.head.weight, 
                targets,
                dynamic_l2warp  # L2 warp factor
            )
            return loss
        else:
            # Original implementation: compute logits, then CE, then L2Wrap
            logits = self(idx)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)), targets.view(-1))
            return L2Wrap.apply(loss, logits)

    def training_step_end(self, batch_parts):
        all = self.all_gather(batch_parts)
        if self.trainer.is_global_zero:
            self.trainer.my_loss_all = all

    def generate_init_weight(self):
        print(
            """
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        n_params = 0
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            s0 = str(shape[0]) if len(shape) > 0 else ""
            s1 = str(shape[1]) if len(shape) > 1 else ""
            s2 = str(shape[2]) if len(shape) > 2 else ""
            s3 = str(shape[3]) if len(shape) > 3 else ""
            print(
                f"{s0.ljust(5)} {s1.ljust(5)} {s2.ljust(5)} {s3.ljust(5)} {n}", end=""
            )

            scale = 1.0
            if (
                "ln_" in n
                or ".ln" in n
                or "time_" in n
                or "_mask" in n
                or "pos_emb" in n
                or ".mask." in n
                or n.endswith("_w")
                or n.endswith("_w1")
                or n.endswith("_w2")
                or n.endswith("_bias")
                or (".weight" not in n)
            ):
                if "ln_x.weight" in n:
                    layer_scale = (
                        1 + int(n.split(".")[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale**0.7)
                else:
                    m[n] = p
                print()
            elif n == "emb.weight":
                m[n] = p
                scale = -1e-4
                nn.init.uniform_(m[n], a=scale, b=-scale)
                print(f" [scale {scale}]")
            elif n == "head.weight":
                m[n] = p
                if self.args.vocab_size > self.args.n_embd:
                    scale = 0.5 * \
                        math.sqrt(self.args.vocab_size / self.args.n_embd)
                else:
                    scale = 0.5
                nn.init.orthogonal_(m[n], gain=scale)
                print(f" [scale {scale}]")
            else:
                assert n.endswith(".weight")  # should always be true

                zero = [
                    ".att.output.",
                    ".ffn.value.",
                    ".ffn.receptance.",
                    ".ffnPre.value.",
                    ".ffnPre.receptance.",
                    "head_q.",
                    ".oo.",
                    ".rr.",
                ]

                for kk in zero:
                    if kk in n:
                        scale = 0

                for kk in [".att.key."]:
                    if kk in n:
                        scale = 0.1
                for kk in [".att.gate."]:
                    if kk in n:
                        scale = 0.1

                print(f" [scale {scale}]")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()
            n_params += m[n].numel()

        print("model params", n_params)
        gc.collect()
        torch.cuda.empty_cache()
        return m
