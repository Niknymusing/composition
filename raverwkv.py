import math
from time import time
from typing import Callable, Dict, Optional
from types import SimpleNamespace
import sys
sys.path.append('/home/nikny/composition/composition/RAVE/')
from types import SimpleNamespace as sn
import gin
import numpy as np
import pytorch_lightning as pl

from pytorch_lightning.utilities import rank_zero_info, rank_zero_only
from pytorch_lightning.strategies import DeepSpeedStrategy

import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam, FusedAdam


import torch
import torch.nn as nn
from einops import rearrange
from sklearn.decomposition import PCA

import rave.core
from .pqmf import CachedPQMF
from . import blocks
from RWKVv4neo import rwkv_model_for_rave as rwkv


class Profiler:

    def __init__(self):
        self.ticks = [[time(), None]]

    def tick(self, msg):
        self.ticks.append([time(), msg])

    def __repr__(self):
        rep = 80 * "=" + "\n"
        for i in range(1, len(self.ticks)):
            msg = self.ticks[i][1]
            ellapsed = self.ticks[i][0] - self.ticks[i - 1][0]
            rep += msg + f": {ellapsed*1000:.2f}ms\n"
        rep += 80 * "=" + "\n\n\n"
        return rep


class WarmupCallback(pl.Callback):

    def __init__(self) -> None:
        super().__init__()
        self.state = {'training_steps': 0}

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        if self.state['training_steps'] >= pl_module.warmup:
            pl_module.warmed_up = True
        self.state['training_steps'] += 1

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


class QuantizeCallback(WarmupCallback):

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:

        if pl_module.warmup_quantize is None: return

        if self.state['training_steps'] >= pl_module.warmup_quantize:
            if isinstance(pl_module.encoder, blocks.DiscreteEncoder):
                pl_module.encoder.enabled = torch.tensor(1).type_as(
                    pl_module.encoder.enabled)
        self.state['training_steps'] += 1


@gin.configurable
class BetaWarmupCallback(pl.Callback):

    def __init__(self, initial_value: float, target_value: float,
                 warmup_len: int) -> None:
        super().__init__()
        self.state = {'training_steps': 0}
        self.warmup_len = warmup_len
        self.initial_value = initial_value
        self.target_value = target_value

    def on_train_batch_start(self, trainer, pl_module, batch,
                             batch_idx) -> None:
        self.state['training_steps'] += 1
        if self.state["training_steps"] >= self.warmup_len:
            pl_module.beta_factor = self.target_value
            return

        warmup_ratio = self.state["training_steps"] / self.warmup_len

        beta = math.log(self.initial_value) * (1 - warmup_ratio) + math.log(
            self.target_value) * warmup_ratio
        pl_module.beta_factor = math.exp(beta)

    def state_dict(self):
        return self.state.copy()

    def load_state_dict(self, state_dict):
        self.state.update(state_dict)


@gin.configurable
class RAVE_RWKV(pl.LightningModule):

    def __init__(
        self,
        input_size,
        latent_size,
        sampling_rate,
        encoder,
        decoder,
        rwkv_class,
        discriminator,
        phase_1_duration,
        gan_loss,
        valid_signal_crop,
        feature_matching_fun,
        num_skipped_features,
        audio_distance: Callable[[], nn.Module],
        multiband_audio_distance: Callable[[], nn.Module],
        weights: Dict[str, float],
        warmup_quantize: Optional[int] = None,
        pqmf: Optional[Callable[[], nn.Module]] = None,
        update_discriminator_every: int = 2,
        enable_pqmf_encode: bool = True,
        enable_pqmf_decode: bool = True,
        # RWKV specific arguments
        FLOAT_MODE='fl32',
        RUN_DEVICE='cuda',
        n_embd=256,
        n_layer=32,
        vocab_size=128,
        my_pos_emb=0,
        pre_ffn=0,
        ctx_len=128,
        dropout=0.0,
        head_qk=0,
        lr_init=0.001,
        accelerator='GPU',
        grad_cp=0,
        weight_decay = 0,
        layerwise_lr=0, 
        my_pile_stage=1,
        betas=(0.9, 0.999),
        adam_eps=1e-8
    ):
        super().__init__()
        #self.device = torch.device('cuda')
        # RAVE components initialization
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.pqmf = pqmf().to(device)#.to(self.device) if pqmf is not None else None
        self.encoder = encoder().to(device)#.to(self.device)
        self.decoder = decoder().to(device)#.to(self.device)
        self.discriminator = discriminator().to(device)#.to(self.device)

        # RWKV arguments wrapped in SimpleNamespace
        self.args = SimpleNamespace(
            FLOAT_MODE=FLOAT_MODE,
            RUN_DEVICE=RUN_DEVICE,
            n_embd=n_embd,
            n_layer=n_layer,
            vocab_size=vocab_size,
            my_pos_emb=my_pos_emb,
            pre_ffn=pre_ffn,
            ctx_len=ctx_len,
            dropout=dropout,
            head_qk=head_qk,
            lr_init=lr_init,
            accelerator=accelerator,
            grad_cp=grad_cp,
            weight_decay=weight_decay, 
            layerwise_lr=layerwise_lr, 
            my_pile_stage=my_pile_stage, 
            betas=betas, 
            adam_eps=adam_eps

        )
        
        # RWKV initialization with SimpleNamespace arguments
        self.rwkv = rwkv_class(self.args).to(device) #.to(self.device)
        self.discriminator = discriminator().to(device)#.to(self.device)

        self.audio_distance = audio_distance()
        self.multiband_audio_distance = multiband_audio_distance()

        self.gan_loss = gan_loss

        self.register_buffer("latent_pca", torch.eye(latent_size))
        self.register_buffer("latent_mean", torch.zeros(latent_size))
        self.register_buffer("fidelity", torch.zeros(latent_size))

        self.input_size = input_size
        self.latent_size = latent_size

        self.automatic_optimization = False

        # SCHEDULE
        self.warmup = phase_1_duration
        self.warmup_quantize = warmup_quantize
        self.weights = weights

        self.warmed_up = False

        # CONSTANTS
        self.sr = sampling_rate
        self.valid_signal_crop = valid_signal_crop
        self.feature_matching_fun = feature_matching_fun
        self.num_skipped_features = num_skipped_features
        self.update_discriminator_every = update_discriminator_every

        self.eval_number = 0
        self.beta_factor = 1.
        self.integrator = None

        self.enable_pqmf_encode = enable_pqmf_encode
        self.enable_pqmf_decode = enable_pqmf_decode

        self.register_buffer("receptive_field", torch.tensor([0, 0]).long())
        


        x = torch.rand(1, 1, 2**17).to(device)
        x = self.pqmf(x)
        z = self.encoder.reparametrize(self.encoder(x))[:1][0]
        self.embedding_sequence_length = z.shape[2]
        z = z.view(1, 1, -1)
        z = self.rwkv(z)
        print('z.shape after rwkv applied : ', z.shape)
        print('self.embedding_sequence_length ', self.embedding_sequence_length)
        print('self.encoder.reparametrize(self.encoder(x))[:1][0].view(1, 1, -1) shape : ', z.shape) 
        print('self.encoder.reparametrize(self.encoder(x))[:1][0].shape : ', self.encoder.reparametrize(self.encoder(x))[:1][0].shape)
        #print('self.encoder.reparametrize(self.encoder(x)).shape : ', self.encoder.reparametrize(self.encoder(x)).shape)
        
        s = self.input_size  # should be 
        print('s : ', type(s), s)
        #x = torch.rand(1, 1, self.input_size)#4096)self.embedding_sequence_length = 
        #self.embedding_sequence_length = self.encoder(self.pqmf(x)).shape[2]
        #print('self.embedding_sequence_length :', self.embedding_sequence_length, 'enc shape 0:', self.encoder(self.pqmf(self.testx)).shape[0],  'enc shape 1:', self.encoder(self.pqmf(self.testx)).shape[1])
        #print('self.input_size :', self.input_size)
        #print('self.latent_size :', self.latent_size)
        #self.automatic_optimization = False
        print('init complete ?')

    def configure_optimizers(self):
        gen_p = list(self.encoder.parameters())
        gen_p += list(self.decoder.parameters())
        dis_p = list(self.discriminator.parameters())

        gen_opt = torch.optim.Adam(gen_p, 1e-4, (.5, .9))
        dis_opt = torch.optim.Adam(dis_p, 1e-4, (.5, .9))


#############################################################################

        #RWKA optimizers below, RAVE optimizers above

#############################################################################


        args = self.args
        
        lr_decay = set()
        lr_1x = set()
        lr_2x = set()
        lr_3x = set()
        for n, p in self.named_parameters():
            if ("time_mix" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_decay" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_3x.add(n)
                else:
                    lr_2x.add(n)
            elif ("time_faaaa" in n) and (args.layerwise_lr > 0):
                if args.my_pile_stage == 2:
                    lr_2x.add(n)
                else:
                    lr_1x.add(n)
            elif ("time_first" in n) and (args.layerwise_lr > 0):
                lr_3x.add(n)
            elif (len(p.squeeze().shape) >= 2) and (args.weight_decay > 0):
                lr_decay.add(n)
            else:
                lr_1x.add(n)

        lr_decay = sorted(list(lr_decay))
        lr_1x = sorted(list(lr_1x))
        lr_2x = sorted(list(lr_2x))
        lr_3x = sorted(list(lr_3x))
        # print('decay', lr_decay)
        # print('1x', lr_1x)
        # print('2x', lr_2x)
        # print('3x', lr_3x)
        param_dict = {n: p for n, p in self.named_parameters()}
        
        if args.layerwise_lr > 0:
            if args.my_pile_stage == 2:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 2e-3 / args.lr_init},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 5.0},# test: 3e-3 / args.lr_init},
                ]
            else:
                optim_groups = [
                    {"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0},
                    {"params": [param_dict[n] for n in lr_2x], "weight_decay": 0.0, "my_lr_scale": 2.0},
                    {"params": [param_dict[n] for n in lr_3x], "weight_decay": 0.0, "my_lr_scale": 3.0},
                ]
        else:
            optim_groups = [{"params": [param_dict[n] for n in lr_1x], "weight_decay": 0.0, "my_lr_scale": 1.0}]

        if args.weight_decay > 0:
            optim_groups += [{"params": [param_dict[n] for n in lr_decay], "weight_decay": args.weight_decay, "my_lr_scale": 1.0}]
            if self.deepspeed_offload:
                return gen_opt, dis_opt#, DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=True, amsgrad=False)
            return gen_opt, dis_opt#, FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=True, amsgrad=False)
        else:
            if self.deepspeed_offload:
                return gen_opt, dis_opt#, DeepSpeedCPUAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adamw_mode=False, weight_decay=0, amsgrad=False)
            return gen_opt, dis_opt#, FusedAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, adam_w_mode=False, weight_decay=0, amsgrad=False)
        # return ZeroOneAdam(optim_groups, lr=self.args.lr_init, betas=self.args.betas, eps=self.args.adam_eps, bias_correction=True, weight_decay=0, amsgrad=False, cuda_aware=False)

    @property
    def deepspeed_offload(self) -> bool:
        strategy = self.trainer.strategy
        if isinstance(strategy, DeepSpeedStrategy):
            cfg = strategy.config["zero_optimization"]
            return cfg.get("offload_optimizer") or cfg.get("offload_param")
        return False

    def split_features(self, features):
        feature_real = []
        feature_fake = []
        for scale in features:
            true, fake = zip(*map(
                lambda x: torch.split(x, x.shape[0] // 2, 0),
                scale,
            ))
            feature_real.append(true)
            feature_fake.append(fake)

        return feature_real, feature_fake

    #def training_step(self, batch, batch_idx):

    def training_step(self, batch, batch_idx):
        print("Batch type:", type(batch))
        print("Batch length:", len(batch))
        if isinstance(batch, list):
            print("First element type:", type(batch[0]))
            print("First element shape:", batch[0].shape)

        p = Profiler()
        gen_opt, dis_opt = self.optimizers()
        x, y = batch#x = batch.unsqueeze(1)

        if self.pqmf is not None:
            x_multiband = self.pqmf(x)
        else:
            x_multiband = x
        p.tick('decompose')

        self.encoder.set_warmed_up(self.warmed_up)
        self.decoder.set_warmed_up(self.warmed_up)

        # ENCODE INPUT
        if self.enable_pqmf_encode:
            z_pre_reg = self.encoder(x_multiband)
        else:
            z_pre_reg = self.encoder(x)

        z, reg = self.encoder.reparametrize(z_pre_reg)[:2]
        p.tick('encode')
        
        #insert RWKV here
        
        # DECODE LATENT
        y_multiband = self.decoder(z)

        p.tick('decode')

        if self.valid_signal_crop and self.receptive_field.sum():
            x_multiband = rave.core.valid_signal_crop(
                x_multiband,
                *self.receptive_field,
            )
            y_multiband = rave.core.valid_signal_crop(
                y_multiband,
                *self.receptive_field,
            )
        p.tick('crop')

        # DISTANCE BETWEEN INPUT AND OUTPUT
        distances = {}

        if self.pqmf is not None:
            multiband_distance = self.multiband_audio_distance(
                x_multiband, y_multiband)
            p.tick('mb distance')

            x = self.pqmf.inverse(x_multiband)
            y = self.pqmf.inverse(y_multiband)
            p.tick('recompose')

            for k, v in multiband_distance.items():
                distances[f'multiband_{k}'] = v
        else:
            x = x_multiband
            y = y_multiband

        fullband_distance = self.audio_distance(x, y)
        p.tick('fb distance')

        for k, v in fullband_distance.items():
            distances[f'fullband_{k}'] = v

        feature_matching_distance = 0.

        if self.warmed_up:  # DISCRIMINATION
            xy = torch.cat([x, y], 0)
            features = self.discriminator(xy)

            feature_real, feature_fake = self.split_features(features)

            loss_dis = 0
            loss_adv = 0

            pred_real = 0
            pred_fake = 0

            for scale_real, scale_fake in zip(feature_real, feature_fake):
                current_feature_distance = sum(
                    map(
                        self.feature_matching_fun,
                        scale_real[self.num_skipped_features:],
                        scale_fake[self.num_skipped_features:],
                    )) / len(scale_real[self.num_skipped_features:])

                feature_matching_distance = feature_matching_distance + current_feature_distance

                _dis, _adv = self.gan_loss(scale_real[-1], scale_fake[-1])

                pred_real = pred_real + scale_real[-1].mean()
                pred_fake = pred_fake + scale_fake[-1].mean()

                loss_dis = loss_dis + _dis
                loss_adv = loss_adv + _adv

            feature_matching_distance = feature_matching_distance / len(
                feature_real)

        else:
            pred_real = torch.tensor(0.).to(x)
            pred_fake = torch.tensor(0.).to(x)
            loss_dis = torch.tensor(0.).to(x)
            loss_adv = torch.tensor(0.).to(x)
        p.tick('discrimination')

        # COMPOSE GEN LOSS
        loss_gen = {}
        loss_gen.update(distances)
        p.tick('update loss gen dict')

        if reg.item():
            loss_gen['regularization'] = reg * self.beta_factor

        if self.warmed_up:
            loss_gen['feature_matching'] = feature_matching_distance
            loss_gen['adversarial'] = loss_adv

        # OPTIMIZATION
        if not (batch_idx %
                self.update_discriminator_every) and self.warmed_up:
            dis_opt.zero_grad()
            loss_dis.backward()
            dis_opt.step()
            p.tick('dis opt')
        else:
            gen_opt.zero_grad()
            loss_gen_value = 0.
            for k, v in loss_gen.items():
                loss_gen_value += v * self.weights.get(k, 1.)
            loss_gen_value.backward()
            gen_opt.step()

        # LOGGING
        self.log("beta_factor", self.beta_factor)

        if self.warmed_up:
            self.log("loss_dis", loss_dis)
            self.log("pred_real", pred_real.mean())
            self.log("pred_fake", pred_fake.mean())

        self.log_dict(loss_gen)
        p.tick('logging')

    def encode(self, x):
        if self.pqmf is not None and self.enable_pqmf_encode:
            x = self.pqmf(x)
        z, = self.encoder.reparametrize(self.encoder(x))[:1]
        z = self.rwkv(z.view(1, 1, -1))
        return z.view(1, -1, self.embedding_sequence_length)

    def decode(self, z):
        y = self.decoder(z)
        if self.pqmf is not None and self.enable_pqmf_decode:
            y = self.pqmf.inverse(y)
        return y

    def forward(self, x):
        #print('self.input_size :', self.input_size)
        #print('self.latent_size :', self.latent_size)
        return self.decode(self.encode(x))

    def validation_step(self, batch, batch_idx):
        x = batch.unsqueeze(1)

        if self.pqmf is not None:
            x_multiband = self.pqmf(x)

        if self.enable_pqmf_encode:
            z = self.encoder(x_multiband)

        else:
            z = self.encoder(x)

        if isinstance(self.encoder, blocks.VariationalEncoder):
            mean = torch.split(z, z.shape[1] // 2, 1)[0]
        else:
            mean = None

        z = self.encoder.reparametrize(z)[0]

        y = self.decoder(z)

        if self.pqmf is not None:
            x = self.pqmf.inverse(x_multiband)
            y = self.pqmf.inverse(y)

        distance = self.audio_distance(x, y)

        full_distance = sum(distance.values())

        if self.trainer is not None:
            self.log('validation', full_distance)

        return torch.cat([x, y], -1), mean

    def on_validation_epoch_end(self, out):
        if not self.receptive_field.sum():
            print("Computing receptive field for this configuration...")
            lrf, rrf = rave.core.get_rave_receptive_field(self)
            self.receptive_field[0] = lrf
            self.receptive_field[1] = rrf
            print(
                f"Receptive field: {1000*lrf/self.sr:.2f}ms <-- x --> {1000*rrf/self.sr:.2f}ms"
            )

        if not len(out): return

        audio, z = list(zip(*out))
        audio = list(map(lambda x: x.cpu(), audio))

        # LATENT SPACE ANALYSIS
        if not self.warmed_up and isinstance(self.encoder,
                                             blocks.VariationalEncoder):
            z = torch.cat(z, 0)
            z = rearrange(z, "b c t -> (b t) c")

            self.latent_mean.copy_(z.mean(0))
            z = z - self.latent_mean

            pca = PCA(z.shape[-1]).fit(z.cpu().numpy())

            components = pca.components_
            components = torch.from_numpy(components).to(z)
            self.latent_pca.copy_(components)

            var = pca.explained_variance_ / np.sum(pca.explained_variance_)
            var = np.cumsum(var)

            self.fidelity.copy_(torch.from_numpy(var).to(self.fidelity))

            var_percent = [.8, .9, .95, .99]
            for p in var_percent:
                self.log(
                    f"fidelity_{p}",
                    np.argmax(var > p).astype(np.float32),
                )

        y = torch.cat(audio, 0)[:8].reshape(-1).numpy()

        if self.integrator is not None:
            y = self.integrator(y)

        self.logger.experiment.add_audio("audio_val", y, self.eval_number,
                                         self.sr)
        self.eval_number += 1

    def on_fit_start(self):
        tb = self.logger.experiment

        config = gin.operative_config_str()
        config = config.split('\n')
        config = ['```'] + config + ['```']
        config = '\n'.join(config)
        tb.add_text("config", config)

        model = str(self)
        model = model.split('\n')
        model = ['```'] + model + ['```']
        model = '\n'.join(model)
        tb.add_text("model", model)
