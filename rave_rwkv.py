import sys
sys.path.append('/home/nikny/composition/composition/RAVE/')
#sys.path.append('/home/nikny/composition/composition')
from rave.model import RAVE
from rave.blocks import EncoderV2, WasserteinEncoder, VariationalEncoder, GeneratorV2
from rave.pqmf import CachedPQMF
#from rwkv.model import RWKV
from rwkv_pip_package.src.rwkv.model import RWKV as RWKV_inference
import torch
from torch import nn
import time
import os
import time

float_mode = 'fl32' #'bf16'

os.environ["RWKV_JIT_ON"] = "1"
os.environ["RWKV_T_MAX"] = "256"
os.environ["RWKV_FLOAT_MODE"] = float_mode
from src.model import RWKV as RWKV_train
from types import SimpleNamespace as sn

args = sn(FLOAT_MODE = float_mode, RUN_DEVICE='cuda', n_embd=512, n_layer=24, 
          vocab_size = 128, my_pos_emb=0, pre_ffn = 0, ctx_len = 128, dropout = 0, 
          head_qk=0, lr_init = 0.001, accelerator = 'GPU', grad_cp=0)
model_train = RWKV_train(args)
model_train.generate_init_weight()
torch.save(model_train.state_dict(), f='rwkv_statedict.pth')
model_inference = RWKV_inference(model='rwkv_statedict.pth', strategy='cuda fp16')#'cuda fp16') # 'cpu fp32')

device = torch.device('cuda:0')#'cpu') #'cuda:0'

def reparametrize(z):
        mean, scale = z.chunk(2, 1)
        std = nn.functional.softplus(scale) + 1e-4
        var = std * std
        logvar = torch.log(var)

        z = torch.randn_like(mean) * std + mean
        kl = (mean * mean + var - logvar - 1).sum(1).mean()

        return z, kl

KERNEL_SIZE = 3
DILATIONS = [
    [1, 3, 9],
    [1, 3, 9],
    [1, 3, 9],
    [1, 3],
]
RATIOS = [4, 4, 4, 2]
CAPACITY = 96
NOISE_AUGMENTATION = 0
LATENT_SIZE = 256
N_BAND = 16

pqmf = CachedPQMF(n_band = N_BAND, attenuation = 82)
encoder = EncoderV2(data_size = N_BAND, capacity = CAPACITY, ratios = RATIOS, 
                    latent_size = LATENT_SIZE, n_out = 2, kernel_size = KERNEL_SIZE, 
                    dilations = DILATIONS) 

generator = GeneratorV2(
    data_size = N_BAND,
    capacity = CAPACITY,
    ratios = RATIOS,
    latent_size = LATENT_SIZE,
    kernel_size = KERNEL_SIZE,
    dilations = DILATIONS,
    amplitude_modulation = True)

encoder.to(device)
generator.to(device)
pqmf.to(device)
model_train.to(device)
model_inference.to(device)

def encode(x):
    z,  = reparametrize(encoder(pqmf(x)))[:1]
    return z

x = 0.001*torch.ones(1, 1, 4096).to(device)
z1 = encode(x)
print('encoding shape = ',z1.shape)
z2, s = model_inference.forward(z1.view(1, 1, -1), None)
print('rwkv output shape =', z2.shape, 'type : ', z2[0][0][0])
print(z2)

y = generator(z2.view(1, -1, 2)).view(1, 1, -1)
print('output shape = ', y.shape)

print(y)



def rwkv_rave(x, s):
    z = encode(x).view(1, 1, -1)
    z, s = model_inference.forward(z, s)
    y = generator(z.view(1, -1, 2))
    return y.view(1, 1, -1), s
t = time.time() 
y, s = rwkv_rave(x, None)
t = time.time() - t
print(y, 'inference time =', t)

class RAVE_RWKV():
     def __init__(self, encoder, rwkv, decoder):
        self.encoder = encoder
        self.rwkv = rwkv
        self.decoder = decoder


y = model_train.forward(z1.view(1, 1, -1))
print('number of rwkv parameters = ', sum(p.numel() for p in model_train.parameters() if p.requires_grad))
t=time.time()
y = model_train(z1.view(1, 1, -1))
t=time.time()-t
print(y)
print('inference time train model = ', t)
