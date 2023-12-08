import gin
import os, math, gc, importlib
import torch
from types import SimpleNamespace 


# torch._C._jit_set_profiling_executor(True)
# torch._C._jit_set_profiling_mode(True)
import torch.nn as nn
from torch.nn import functional as F
import pytorch_lightning as pl
from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy
if importlib.util.find_spec('deepspeed'):
    import deepspeed
    from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam

# from deepspeed.runtime.fp16.onebit.zoadam import ZeroOneAdam

try:
    print('RWKV_MY_TESTING', os.environ["RWKV_MY_TESTING"])
except:
    os.environ["RWKV_MY_TESTING"] = ''

def __nop(ob):
    return ob


MyModule = nn.Module
MyFunction = __nop
if os.environ["RWKV_JIT_ON"] == "1":
    MyModule = torch.jit.ScriptModule
    MyFunction = torch.jit.script_method


########################################################################################################
# CUDA Kernel
########################################################################################################

T_MAX = int(os.environ["RWKV_T_MAX"])  # TAKES LOTS OF VRAM!
# it's possible to go beyond CUDA limitations if you slice the ctx and pass the hidden state in each slice

from torch.utils.cpp_extension import load

if os.environ["RWKV_FLOAT_MODE"] == "bf16":
    wkv_cuda = load(name=f"wkv_{T_MAX}_bf16", sources=["cuda/wkv_op_bf16.cpp", "cuda/wkv_cuda_bf16.cu"], verbose=True, extra_cuda_cflags=["-t 4", "-std=c++17", "-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w = -torch.exp(w.float().contiguous())
            u = u.contiguous().to(dtype=torch.bfloat16)
            k = k.contiguous().to(dtype=torch.bfloat16)
            v = v.contiguous().to(dtype=torch.bfloat16)
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            return y
        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format, dtype=torch.bfloat16)
            wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            return (None, None, None, gw, gu, gk, gv)
else:
    wkv_cuda = load(name=f"wkv_{T_MAX}", sources=["cuda/wkv_op.cpp", "cuda/wkv_cuda.cu"], verbose=True, extra_cuda_cflags=["-res-usage", "--maxrregcount 60", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-DTmax={T_MAX}"])
    class WKV(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, w, u, k, v):
            ctx.B = B
            ctx.T = T
            ctx.C = C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                w = -torch.exp(w.contiguous())
                u = u.contiguous()
                k = k.contiguous()
                v = v.contiguous()
            else:
                w = -torch.exp(w.float().contiguous())
                u = u.float().contiguous()
                k = k.float().contiguous()
                v = v.float().contiguous()
            y = torch.empty((B, T, C), device=w.device, memory_format=torch.contiguous_format)
            wkv_cuda.forward(B, T, C, w, u, k, v, y)
            ctx.save_for_backward(w, u, k, v, y)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return y
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return y.half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return y.bfloat16()
        @staticmethod
        def backward(ctx, gy):
            B = ctx.B
            T = ctx.T
            C = ctx.C
            assert T <= T_MAX
            assert B * C % min(C, 32) == 0
            w, u, k, v, y = ctx.saved_tensors
            gw = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gu = torch.empty((B, C), device=gy.device, memory_format=torch.contiguous_format)
            gk = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            gv = torch.empty((B, T, C), device=gy.device, memory_format=torch.contiguous_format)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.contiguous(), gw, gu, gk, gv)
            else:
                wkv_cuda.backward(B, T, C, w, u, k, v, y, gy.float().contiguous(), gw, gu, gk, gv)
            gw = torch.sum(gw, dim=0)
            gu = torch.sum(gu, dim=0)
            if "32" in os.environ["RWKV_FLOAT_MODE"]:
                return (None, None, None, gw, gu, gk, gv)
            elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
                return (None, None, None, gw.half(), gu.half(), gk.half(), gv.half())
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                return (None, None, None, gw.bfloat16(), gu.bfloat16(), gk.bfloat16(), gv.bfloat16())


def RUN_CUDA(B, T, C, w, u, k, v):
    return WKV.apply(B, T, C, w, u, k, v)

########################################################################################################

class RWKV_TimeMix_RWKV5_Preview(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = 64
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = 8

        self.chunk_len = 512
        assert args.ctx_len % self.chunk_len == 0

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            if 'r3' in os.environ["RWKV_MY_TESTING"]:
                self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
                self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)

            # fancy time_decay
            decay_speed = torch.ones(self.n_head)
            for h in range(self.n_head):
                decay_speed[h] = -6 + 5 * (h / (self.n_head - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            if 'r2' in os.environ["RWKV_MY_TESTING"]:
                tmp = torch.zeros(self.n_head)
                for h in range(self.n_head):
                    tmp[h] = ratio_0_to_1 * (1 - (h / (self.n_head - 1)))
                self.time_faaaa = nn.Parameter(tmp)
            else:
                self.time_first = nn.Parameter(torch.ones(self.n_head) * (-3.0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    if 'r3' in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def jit_func(self, x):
            B, TT, C = x.size()

            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

            r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
            k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
            v = self.value(xv).view(B, TT, self.n_head, -1).transpose(1, 2)                 # BTC -> BHTS
            g = F.silu(self.gate(xg))

            return r, k, v, g

        @MyFunction
        def jit_func_2(self, r, k, v, g, w, wk, wb, ws):
            B, H, TT, S = r.size()
            T = self.chunk_len

            s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
            x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

            for i in range(TT // T):
                rr = r[:, :, i*T:i*T+T, :]
                kk = k[:, :, :, i*T:i*T+T]
                vv = v[:, :, i*T:i*T+T, :]

                x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

                s = ws * s + (kk * wk) @ vv
            
            x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
            x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S) * g
            return self.output(x)
    else:
        @MyFunction
        def jit_func(self, x):
            B, TT, C = x.size()

            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)

            r = self.receptance(xr).view(B, TT, self.n_head, self.head_size).transpose(1, 2)            # BTC -> BHTS
            k = self.key(xk).view(B, TT, self.n_head, self.head_size).transpose(1, 2).transpose(-2, -1) # BTC -> BHTS -> BHST
            v = self.value(xv).view(B, TT, self.n_head, self.head_size).transpose(1, 2)                 # BTC -> BHTS

            return r, k, v

        @MyFunction
        def jit_func_2(self, r, k, v, w, wk, wb, ws):
            B, H, TT, S = r.size()
            T = self.chunk_len

            s = torch.zeros(B, H, S, S, device=r.device, dtype=r.dtype)  # state
            x = torch.zeros(B, H, TT, S, device=r.device, dtype=r.dtype) # output

            for i in range(TT // T):
                rr = r[:, :, i*T:i*T+T, :]
                kk = k[:, :, :, i*T:i*T+T]
                vv = v[:, :, i*T:i*T+T, :]

                x[:, :, i*T:i*T+T, :] = ((rr @ kk) * w) @ vv  +  (rr @ s) * wb

                s = ws * s + (kk * wk) @ vv
            
            x = x.transpose(1, 2).contiguous().view(B * TT, H*S) # BHTS -> BTHS -> BTC
            x = self.ln_x(x / self.head_size_divisor).view(B, TT, H*S)
            return self.output(x)
    
    def forward(self, x):
        H = self.n_head
        T = self.chunk_len

        if 'r3' in os.environ["RWKV_MY_TESTING"]:
            r, k, v, g = self.jit_func(x)
        else:
            r, k, v = self.jit_func(x)

        w = torch.exp(-torch.exp(self.time_decay.float())).unsqueeze(-1)
        
        if 'r2' in os.environ["RWKV_MY_TESTING"]:
            u = self.time_faaaa.float().unsqueeze(-1)
        else:
            u = torch.exp(self.time_first.float()).unsqueeze(-1)

################################################################################
########
        ws = w.pow(T).reshape(1, H, 1, 1)

        ind = torch.arange(T-1, -1, -1, device=r.device).unsqueeze(0).repeat(H, 1)
        w = w.repeat(1, T).pow(ind)

        wk = w.reshape(1, H, 1, T)
        wb = wk.transpose(-2, -1).flip(2)

        w = torch.cat([w[:, 1:], u], dim=1)
        w = F.pad(w, (0, T))
        w = torch.tile(w, [T])
        w = w[:, :-T].reshape(-1, T, 2 * T - 1)
        w = w[:, :, T-1:].reshape(1, H, T, T)
########
################################################################################

        w = w.to(dtype=r.dtype)
        wk = wk.to(dtype=r.dtype)
        wb = wb.to(dtype=r.dtype)
        ws = ws.to(dtype=r.dtype)
        if 'r3' in os.environ["RWKV_MY_TESTING"]:
            return self.jit_func_2(r, k, v, g, w, wk, wb, ws)
        else:
            return self.jit_func_2(r, k, v, w, wk, wb, ws)        


########################################################################################################
# CUDA RWKV5 Kernel
########################################################################################################

if 'r4' in os.environ["RWKV_MY_TESTING"]:
    HEAD_SIZE = int(os.environ["RWKV_HEAD_SIZE_A"])
    wkv5_cuda = load(name="wkv5", sources=["cuda/wkv5_op.cpp", f"cuda/wkv5_cuda.cu"],
                    verbose=True, extra_cuda_cflags=["-res-usage", "--use_fast_math", "-O3", "-Xptxas -O3", "--extra-device-vectorization", f"-D_N_={HEAD_SIZE}"])
        
    class WKV_5(torch.autograd.Function):
        @staticmethod
        def forward(ctx, B, T, C, H, r, k, v, w, u):
            with torch.no_grad():
                assert r.dtype == torch.bfloat16
                assert k.dtype == torch.bfloat16
                assert v.dtype == torch.bfloat16
                assert w.dtype == torch.bfloat16
                assert u.dtype == torch.bfloat16
                assert HEAD_SIZE == C // H
                ctx.B = B
                ctx.T = T
                ctx.C = C
                ctx.H = H
                assert r.is_contiguous()
                assert k.is_contiguous()
                assert v.is_contiguous()
                assert w.is_contiguous()
                assert u.is_contiguous()
                ew = (-torch.exp(w.float())).contiguous()
                eew = (torch.exp(ew)).contiguous()
                ctx.save_for_backward(r, k, v, eew, ew, u)
                y = torch.empty((B, T, C), device=r.device, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.forward(B, T, C, H, r, k, v, eew, u, y)
                return y

        @staticmethod
        def backward(ctx, gy):
            with torch.no_grad():
                assert gy.dtype == torch.bfloat16
                B = ctx.B
                T = ctx.T
                C = ctx.C
                H = ctx.H
                assert gy.is_contiguous()
                r, k, v, eew, ew, u = ctx.saved_tensors
                gr = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gk = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gv = torch.empty((B, T, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gw = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                gu = torch.empty((B, C), device=gy.device, requires_grad=False, dtype=torch.bfloat16, memory_format=torch.contiguous_format) # .uniform_(-1, 1)
                wkv5_cuda.backward(B, T, C, H, r, k, v, eew, ew, u, gy, gr, gk, gv, gw, gu)
                gw = torch.sum(gw, 0).view(H, C//H)
                gu = torch.sum(gu, 0).view(H, C//H)
                return (None, None, None, None, gr, gk, gv, gw, gu)

    def RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w, u):
        return WKV_5.apply(B, T, C, H, r, k, v, w, u)

########################################################################################################

class RWKV_TimeMix_RWKV5(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.head_size = args.head_size_a
        assert HEAD_SIZE == self.head_size # change HEAD_SIZE to match args.head_size_a
        self.n_head = args.dim_att // self.head_size
        assert args.dim_att % self.n_head == 0
        self.head_size_divisor = args.head_size_divisor

        with torch.no_grad():
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))
            self.time_mix_g = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for n in range(args.dim_att):
                decay_speed[n] = -6 + 5 * (n / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed.reshape(self.n_head, self.head_size))
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            tmp = torch.zeros(args.dim_att)
            for n in range(args.dim_att):
                zigzag = ((n + 1) % 3 - 1) * 0.1
                tmp[n] = ratio_0_to_1 * (1 - (n / (args.dim_att - 1))) + zigzag

            self.time_faaaa = nn.Parameter(tmp.reshape(self.n_head, self.head_size))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)

        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)
        self.gate = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.ln_x = nn.GroupNorm(self.n_head, args.dim_att)

    @MyFunction
    def jit_func(self, x):
        B, T, C = x.size()

        xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        xg = x * self.time_mix_g + xx * (1 - self.time_mix_g)

        r = self.receptance(xr)
        k = self.key(xk)
        v = self.value(xv)
        g = F.silu(self.gate(xg))

        return r, k, v, g

    @MyFunction
    def jit_func_2(self, x, g):
        B, T, C = x.size()
        x = x.view(B * T, C)
        
        x = self.ln_x(x / self.head_size_divisor).view(B, T, C)
        x = self.output(x * g)
        return x

    def forward(self, x):
        B, T, C = x.size()
        H = self.n_head

        r, k, v, g = self.jit_func(x)

        x = RUN_CUDA_RWKV5(B, T, C, H, r, k, v, w=self.time_decay, u=self.time_faaaa)

        return self.jit_func_2(x, g)

########################################################################################################
# RWKV: RWKV Time-mix + RWKV Channel-mix
########################################################################################################


class RWKV_TimeMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.ctx_len = args.ctx_len
        self.n_embd = args.n_embd

        with torch.no_grad():  # fancy init
            ratio_0_to_1 = layer_id / (args.n_layer - 1)  # 0 to 1
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            
            # fancy time_decay
            decay_speed = torch.ones(args.dim_att)
            for h in range(args.dim_att):
                decay_speed[h] = -5 + 8 * (h / (args.dim_att - 1)) ** (0.7 + 1.3 * ratio_0_to_1)
            self.time_decay = nn.Parameter(decay_speed)
            # print(layer_id, self.time_decay.flatten()[:3].cpu().numpy(), '...', self.time_decay.flatten()[-3:].cpu().numpy())

            # fancy time_first
            zigzag = torch.tensor([(i + 1) % 3 - 1 for i in range(args.dim_att)]) * 0.5
            self.time_first = nn.Parameter(torch.ones(args.dim_att) * math.log(0.3) + zigzag)

            # fancy time_mix
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_v = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)
            self.time_mix_r = nn.Parameter(torch.pow(ddd, 0.5 * ratio_1_to_almost0))

        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        self.key = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.value = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.dim_att, bias=False)
        self.output = nn.Linear(args.dim_att, args.n_embd, bias=False)

        if 'a' in os.environ["RWKV_MY_TESTING"]:
            self.register_buffer("att_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))
            d_qkv = args.n_embd // 16
            self.qq = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.kk = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.vv = nn.Linear(args.n_embd, d_qkv, bias=False)
            self.oo = nn.Linear(d_qkv, args.n_embd, bias=False)
            with torch.no_grad():
                self.time_mix_qq = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_kk = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
                self.time_mix_vv = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0) + 0.3 * ratio_0_to_1)

    if 'a' not in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def jit_func(self, x):
            #print('x ', x)
            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            #print('xx ', xx)
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            #print('xk ', xk)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            #print('sr ', sr)
            return sr, k, v

        def forward(self, x):
            
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            
            sr, k, v = self.jit_func(x)
            #if self.layer_id ==0:
                #print('nans in any of the following jit_func outputs; ')
            #    l=[sr, k, v]
                #for i in range(len(l)):
                    #print(l[i])
                    #if torch.isnan(l[i]).any():
                        #print(i)
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
            #if self.layer_id ==0:
                #print('rwkv layer 0:')
                #print(rwkv)
            #if torch.isnan(rwkv.any()):
            #        print('nan in rwkv')
            #else:
            #    print('nan not found yet')
            #if torch.isnan(self.output(rwkv).any()):
            #        print('nan in rwkv')
            #else:
            #    print('nan not found yet')
            #print(self.output(rwkv))
            return self.output(rwkv)

    if 'a' in os.environ["RWKV_MY_TESTING"]:
        @MyFunction
        def QKV(self, q, k, v):
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.att_mask == 0, float('-inf'))
            att = F.softmax(att, dim = -1)
            x = att @ v
            return x

        @MyFunction
        def jit_funcQKV(self, x):
            xx = self.time_shift(x) # Mix x with the previous timestep to produce xk, xv, xr
            xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
            xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
            xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
            xqq = x * self.time_mix_qq + xx * (1 - self.time_mix_qq)
            xkk = x * self.time_mix_kk + xx * (1 - self.time_mix_kk)
            xvv = x * self.time_mix_vv + xx * (1 - self.time_mix_vv)
            k = self.key(xk)
            v = self.value(xv)
            r = self.receptance(xr)
            sr = torch.sigmoid(r)
            qq = self.qq(xqq)
            kk = self.kk(xkk)
            vv = self.vv(xvv)
            return sr, k, v, qq, kk, vv

        def forward(self, x):
            B, T, C = x.size()  # x = (Batch,Time,Channel)
            sr, k, v, qq, kk, vv = self.jit_funcQKV(x)
            #l=[sr, k, v, qq, kk, vv]
            #for i in range(len(l)):
            #    print(-i)
            #    if torch.isnan(l[i]):
            #        print(i)
    #                print('torch.isnan([sr, k, v, qq, kk, vv]).any() :', torch.isnan([sr, k, v, qq, kk, vv]).any())
            rwkv = sr * RUN_CUDA(B, T, self.args.dim_att, self.time_decay, self.time_first, k, v)
            rwkv = self.output(rwkv) + self.oo(self.QKV(qq, kk, vv))
            return rwkv

########################################################################################################

class RWKV_ChannelMix(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():  # fancy init of time_mix
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)  # 1 to ~0
            ddd = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                ddd[0, 0, i] = i / args.n_embd
            self.time_mix_k = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(ddd, ratio_1_to_almost0))
        
        self.key = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
        self.receptance = nn.Linear(args.n_embd, args.n_embd, bias=False)
        self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        k = self.key(xk)
        k = torch.relu(k) ** 2
        kv = self.value(k)
        return torch.sigmoid(self.receptance(xr)) * kv

class MishGLU(MyModule):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))

        with torch.no_grad():
            ratio_1_to_almost0 = 1.0 - (layer_id / args.n_layer)

            x = torch.ones(1, 1, args.n_embd)
            for i in range(args.n_embd):
                x[0, 0, i] = i / args.n_embd

            self.time_mix_k = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.time_mix_r = nn.Parameter(torch.pow(x, ratio_1_to_almost0))
            self.aa = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.bb = nn.Linear(args.n_embd, args.dim_ffn, bias=False)
            self.value = nn.Linear(args.dim_ffn, args.n_embd, bias=False)

    @MyFunction
    def forward(self, x):
        xx = self.time_shift(x)
        xa = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xb = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        a = self.aa(xa)
        b = self.bb(xb)
        return self.value(a * F.mish(b))

########################################################################################################
# The RWKV Model with our blocks
########################################################################################################


class Block(nn.Module):
    def __init__(self, args, layer_id):
        super().__init__()
        self.args = args
        self.layer_id = layer_id

        self.ln1 = nn.LayerNorm(args.n_embd)
        self.ln2 = nn.LayerNorm(args.n_embd)

        if self.layer_id == 0:
            self.ln0 = nn.LayerNorm(args.n_embd)
            if args.my_pos_emb > 0:
                self.pos_emb_x = nn.Parameter(torch.zeros((1,args.my_pos_emb,args.n_embd)))
                self.pos_emb_y = nn.Parameter(torch.zeros((args.my_pos_emb,1,args.n_embd)))

        if self.layer_id == 0 and self.args.pre_ffn > 0:
            self.ffnPre = RWKV_ChannelMix(args, 0)
        else:
            if 'r4' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_TimeMix_RWKV5(args, layer_id)
                #print('time mix 5')
            elif 'r' in os.environ["RWKV_MY_TESTING"]:
                self.att = RWKV_TimeMix_RWKV5_Preview(args, layer_id)
                #print('time mix 5prev')
            else:
                self.att = RWKV_TimeMix(args, layer_id)
                #print('time mix regular')

        if 'g' in os.environ["RWKV_MY_TESTING"]:
            self.ffn = MishGLU(args, layer_id)
        else:
            self.ffn = RWKV_ChannelMix(args, layer_id)
        
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            self.tiny_ln = nn.LayerNorm(args.n_embd)
            self.tiny_q = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_k = nn.Linear(args.n_embd, args.tiny_att_dim, bias=False)
            self.tiny_v = nn.Linear(args.n_embd, args.n_embd, bias=False)
            self.register_buffer("tiny_mask", torch.tril(torch.ones(args.ctx_len, args.ctx_len)))

        if args.dropout > 0:
            self.drop0 = nn.Dropout(p = args.dropout)
            self.drop1 = nn.Dropout(p = args.dropout)
        
    def forward(self, x, x_emb=None):
        args = self.args
        B, T, C = x.size()
        
        if self.layer_id == 0:
            x = self.ln0(x)
            
            #print('eyyyyyyyyyyyyyyyyyyyyyyyyy self.ln0(x) ', self.ln0(x))
            #print('in block, line1: x = self.ln0(x),    torch.isnan(x).any() =' ,torch.isnan(x).any())
            if args.my_pos_emb > 0:
                pos_emb = (self.pos_emb_x + self.pos_emb_y).reshape(T+1, -1)[:-1,:]
                x = x + pos_emb
                #print('in block, line2: x = x + pos_emb,    torch.isnan(x).any() =' ,torch.isnan(x).any())

        if self.args.dropout == 0:

            
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = x + self.ffnPre(self.ln1(x))
                
                #print('in block, line3: x = x + self.ffnPre(self.ln1(x)),    torch.isnan(x).any() =' ,torch.isnan(x).any())
                
            else:
                #if self.layer_id == 0:
                    #print('eyyyyyyyyyyyyyyyyyyyyyyyyy 1 ', x)
                    #print('self.ln1(x) ', self.ln1(x))
                
                
                
                x = x + self.att(self.ln1(x))   # conclusive after debugging: there appears nans in self.att() from layer_id 0
                
            x = x + self.ffn(self.ln2(x))
            #print('in block, line5:x = x + self.ffn(self.ln2(x)),    torch.isnan(x).any() =' ,torch.isnan(x).any())
        else:
            if self.layer_id == 0 and args.pre_ffn > 0:
                x = self.drop0(x + self.ffnPre(self.ln1(x)))
            #    print('in block, line6: x = self.drop0(x + self.ffnPre(self.ln1(x))),    torch.isnan(x).any() =' ,torch.isnan(x).any())
            else:
                #print('eyyyyyyyyyyyyyyyyyyyyyyyyy x = x + self.ffnPre(self.ln1(x)) ', x)
                x = self.drop0(x + self.att(self.ln1(x)))
            #    print('in block, line7:x = self.drop0(x + self.att(self.ln1(x))),    torch.isnan(x).any() =' ,torch.isnan(x).any())
            x = self.drop1(x + self.ffn(self.ln2(x)))
            #print('in block, line8: x = self.drop1(x + self.ffn(self.ln2(x))),    torch.isnan(x).any() =' ,torch.isnan(x).any())
        if args.tiny_att_dim > 0 and self.layer_id == args.tiny_att_layer:
            xx = self.tiny_ln(x)
            #print('in block, line9: xx = self.tiny_ln(x),    torch.isnan(x).any() =' ,torch.isnan(xx).any())
            q = self.tiny_q(xx)[:, :T, :]
            k = self.tiny_k(xx)[:, :T, :]
            c = (q @ k.transpose(-2, -1)) * (args.tiny_att_dim ** (-0.5))
            c = c.masked_fill(self.tiny_mask[:T, :T] == 0, 0)
            x = x + c @ self.tiny_v(x_emb)
            #print('eyyyyyyyyyyyyyyyyyyyyyyyyy 2 ', x)
            #print('in block, line10: x = x + c @ self.tiny_v(x_emb),    torch.isnan(x).any() =' ,torch.isnan(x).any())
        #if self.layer_id == 0:
            #print('eyyyyyyyyyyyyyyyyyyyyyyyyy 2 ', x)
        return x


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






@gin.configurable
class RWKVConfig:
    def __init__(self, FLOAT_MODE='fl32', RUN_DEVICE='cuda', n_embd=512, n_layer=24, 
                 vocab_size=128, my_pos_emb=0, pre_ffn=0, ctx_len=128, dropout=0, 
                 head_qk=0, lr_init=0.001, accelerator='GPU', grad_cp=0):
        self.FLOAT_MODE = FLOAT_MODE
        self.RUN_DEVICE = RUN_DEVICE
        self.n_embd = n_embd
        self.n_layer = n_layer
        self.vocab_size = vocab_size
        self.my_pos_emb = my_pos_emb
        self.pre_ffn = pre_ffn
        self.ctx_len = ctx_len
        self.dropout = dropout
        self.head_qk = head_qk
        self.lr_init = lr_init
        self.accelerator = accelerator
        self.grad_cp = grad_cp

# Register RWKVConfig with Gin
gin.external_configurable(RWKVConfig, 'RWKVConfig')



@gin.configurable
class RWKV(nn.Module):
    def __init__(self, args):
        super().__init__()
        # Create a SimpleNamespace from the arguments
        self.args = args
        if not hasattr(self.args, 'dim_att'):
            self.args.dim_att = self.args.n_embd
        if not hasattr(self.args, 'dim_ffn'):
            self.args.dim_ffn =self. args.n_embd * 4
        if not hasattr(self.args, 'tiny_att_layer'):
            self.args.tiny_att_layer = -1
        if not hasattr(self.args, 'tiny_att_dim'):
            self.args.tiny_att_dim = -1
        assert self.args.n_embd % 32 == 0
        assert self.args.dim_att % 32 == 0
        assert self.args.dim_ffn % 32 == 0
        print('modified rwkv initialised')
        #self.emb = nn.Embedding(args.vocab_size, args.n_embd)
        
        self.blocks = nn.ModuleList([Block(self.args, i) for i in range(self.args.n_layer)])
        self.ln_out = nn.LayerNorm(self.args.n_embd)
        #self.head = nn.Linear(args.n_embd, args.vocab_size, bias=False)

        if self.args.head_qk > 0:
            self.head_q = nn.Linear(self.args.n_embd, self.args.head_qk, bias=False)
            self.head_k = nn.Linear(self.args.n_embd, self.args.head_qk, bias=False)
            self.register_buffer("copy_mask", torch.tril(torch.ones(self.args.ctx_len, self.args.ctx_len)))
        if self.args.dropout > 0:
            self.drop0 = nn.Dropout(p = self.args.dropout)
        self.grad_cp = self.args.grad_cp

    def forward(self, x):
        args = self.args
        #print('idx.size() =', idx.size())
        B, T, _ = x.size()
        assert T <= args.ctx_len, "Cannot forward, model ctx_len is exhausted."

        #self.emb(idx)
        x_emb = x
        #print('x_emb.shape = ', x_emb.shape, 'x.shape =', x.shape, 'torch.isnan(x).any() 1 :' ,torch.isnan(x).any())
        #i=0
        if args.dropout > 0:
            x = self.drop0(x)
            #print('torch.isnan(x).any() 2 :' ,torch.isnan(x).any())
        if args.tiny_att_dim > 0:
        
            for block in self.blocks:
                
                if self.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x, x_emb)
                    #print('torch.isnan(x).any() 3 :'+str(i) ,torch.isnan(x).any())
                else:
                    x = block(x, x_emb)
                    #print('torch.isnan(x).any() 4 :'+str(i) ,torch.isnan(x).any())
                
                #i+=1
        else:
            for block in self.blocks:
                if args.grad_cp == 1:
                    x = deepspeed.checkpointing.checkpoint(block, x)
                    #print('torch.isnan(x).any() 5 :'+str(i) ,torch.isnan(x).any())
                else:
                    x = block(x)
                    #print('torch.isnan(x).any() 6 :'+str(i) ,torch.isnan(x).any())
                
                #i+=1

        x = self.ln_out(x)

        #if args.head_qk > 0:
        #    q = self.head_q(x)[:, :T, :]
        #    k = self.head_k(x)[:, :T, :]
        #    c = (q @ k.transpose(-2, -1)) * (1.0 / args.head_qk)
        #    c = c.masked_fill(self.copy_mask[:T, :T] == 0, 0)

        #    if "32" in os.environ["RWKV_FLOAT_MODE"]:
        #        c = c @ F.one_hot(idx, num_classes=args.vocab_size)
        #    elif os.environ["RWKV_FLOAT_MODE"] == "fp16":
        #        c = c @ F.one_hot(idx, num_classes=args.vocab_size).half()
        #    elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
        #        c = c @ F.one_hot(idx, num_classes=args.vocab_size).bfloat16()

        #    x = self.head(x) + c
        #else:
        #    x = self.head(x)

        return x

    

    def generate_init_weight(self):
        print(
            f"""
############################################################################
#
# Init model weight (slow for large models)...
#
############################################################################
"""
        )
        m = {}
        for n in self.state_dict():
            p = self.state_dict()[n]
            shape = p.shape

            gain = 1.0
            scale = 1.0
            if "ln_" in n or ".ln" in n or "time_" in n or "_mask" in n or "pos_emb" in n or '.mask.' in n:
                if 'ln_x.weight' in n:
                    layer_scale = (1+int(n.split('.')[1])) / self.args.n_layer
                    m[n] = (p * 0.0) + (layer_scale ** 0.7)
                else:
                    m[n] = p
            else:
                if n == "emb.weight":
                    scale = -1 * self.args.lr_init
                else:
                    if shape[0] > shape[1]:
                        gain = math.sqrt(shape[0] / shape[1])
                    if 'r' in os.environ["RWKV_MY_TESTING"]:
                        zero = [".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                    else:
                        zero = [".att.key.", ".att.receptance.", ".att.output.", ".ffn.value.", ".ffn.receptance.", ".ffnPre.value.", ".ffnPre.receptance.", "head_q.", '.oo.', '.rr.']
                    for kk in zero:
                        if kk in n:
                            scale = 0
                    if n == "head.weight":
                        scale = 0.5
                    if "head_k." in n:
                        scale = 0.1
                    if "head_q." in n:
                        scale = 0

                print(f"{str(shape[0]).ljust(5)} {str(shape[1]).ljust(5)} {str(scale).ljust(4)} {n}")

                if self.args.accelerator.upper() == "GPU":
                    m[n] = torch.empty((shape[0], shape[1]), device="cuda")
                else:
                    m[n] = torch.empty((shape[0], shape[1]))

                if scale == 0:
                    nn.init.zeros_(m[n])
                elif scale < 0:
                    nn.init.uniform_(m[n], a=scale, b=-scale)
                else:
                    nn.init.orthogonal_(m[n], gain=gain * scale)

            m[n] = m[n].cpu()
            if os.environ["RWKV_FLOAT_MODE"] == "fp16":
                m[n] = m[n].half()
            elif os.environ["RWKV_FLOAT_MODE"] == "bf16":
                m[n] = m[n].bfloat16()

            # if n == "emb.weight":
            #     print(m[n])
            print('initialised and generated weights for modified rwkv')

        gc.collect()
        torch.cuda.empty_cache()
        return m