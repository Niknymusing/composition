import torch
from torch import nn
import cv2
import time
import mediapipe as mp
import numpy as np
import signal
import sys
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam
#from sklearn.model_selection import train_test_split
import numpy as np
from torch import nn
from spiralnet import instantiate_model as instantiate_spiralnet 
from torch.nn import init
from collections import deque
from pythonosc import udp_client
from dancestrument.utils.midi_functions import send_notes
#from dancestrument.utils.ableton_functions import play_a_clip
#from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
#import asyncio
import threading
import rtmidi
from rtmidi.midiconstants import (CONTROL_CHANGE)
import time
import random
from rave.blocks import EncoderV2
from rave.pqmf import CachedPQMF
from multimodal_model import SpiralnetRAVEClassifierGRU
import pyaudio 
import queue               
import sys
import os
sys.path.append(os.getcwd()+'/s4')
from s4.src.models.sequence.modules.s4block import S4Block as S4 
from itertools import product


class AudioEncoder(nn.Module):
    def __init__(self):
        super(AudioEncoder, self).__init__()

        KERNEL_SIZE = 3
        DILATIONS = [
            [1, 3, 9],
            #[1, 3, 9],
            #[1, 3, 9],
            [1, 3],
        ]
        RATIOS = [16, 8]
        CAPACITY = 24#96#64#
        NOISE_AUGMENTATION = 0
        LATENT_SIZE = 16
        N_BAND = LATENT_SIZE# 16

        #self.pqmf = CachedPQMF(n_band = N_BAND, attenuation = 100)
        self.encoder = EncoderV2(data_size = N_BAND, capacity = CAPACITY, ratios = RATIOS, 
                            latent_size = LATENT_SIZE, n_out = 1, kernel_size = KERNEL_SIZE, 
                            dilations = DILATIONS) 
    def forward(self, x):
        return self.encoder(x)


class MultiscaleSequence_MINE_(nn):
    def __init__(self, time_scales_past: list, 
                 time_scales_future: list, 
                 latent_input_dim, 
                 latent_dim 
                 ):
        
        
        # INPUTS: 
        # - embedding vectors for current time step over 4 past time-windows of incr length  time scales,
        # - joint samples of future audio input PQMF embeddings over 4 future time-windows of incr length  time scales
        # - marginal samples of future audio input PQMF embeddings over 4 future time-windows of incr length  time scales
        # OUTPUT: 
        # for each pairs of future and past time scales, estimate MINE by:

        # - 0) jointly embed the latent vectors and the joint resp. marginally distributed audio PQMF, by some simple encoding op, e.g. downsampling CNN. 
        
        # - 1) average pooling downsample so all the latent vectors and PQMF audio embeddings have same tensor shape (1, 16, 128) across all time scales, the embedding 
        
        # - 2) for each pairs (i, j) of past time scale i and future time scale j, input triple (emb_Ti, audio_joint_Tj, audio_marg_Tj) to its own S4 layer
        
        # - 3) multiply the outputs by the learned weightning factor preinitilised as w_ij = right_skewed_normal_[mean = -buffer_size](-i) * left_skewed_normal_[mean = buffer_size](j)
        
        # - 4) project concatenated output of all estimators by linear layer to high-dim space
        
        # - 5) downproject from high dim space to obtain i*j-dim vector with one float value for each of the pairs (i, j)
        
        #qself.s4_layers = nn.ModuleList()
        pass
    def forward(self, audio_emb, audio_joint, audio_marg):
        pass


import torch
import torch.nn as nn
import torch.nn.functional as F


class RunningMineMean:
    def __init__(self):
        self.sum_joints = torch.tensor(0.0)  # Running sum of elements
        self.sum_margs = torch.tensor(0.0)
        self.count = torch.tensor(0.0)  # Count of elements

    def update(self, y_joint, y_marg):
        self.sum_joints += y_joint
        self.sum_margs += y_marg
        self.count += 1
        return self.sum_joints / self.count - torch.log(torch.exp(self.sum_margs / self.count))

    def mine_mean(self):
        if self.count == 0:
            return torch.tensor(float('nan'))  # Handle division by 0
        return self.sum_joints / self.count - torch.log(torch.exp(self.sum_margs / self.count))

    


class MultiscaleSequence_MINE(nn.Module):
    def __init__(self, time_scales_past: list, 
                 time_scales_future: list, 
                 latent_input_dim, 
                 latent_dim):
        super(MultiscaleSequence_MINE, self).__init__()

        self.time_scales_past = time_scales_past
        self.time_scales_future = time_scales_future
        self.nr_past_timescales = len(self.time_scales_past)
        self.nr_future_timescales = len(self.time_scales_future)
        self.latent_input_dim = latent_input_dim
        self.latent_dim = latent_dim
        self.audio_encoder = AudioEncoder()  # Make sure AudioEncoder is defined somewhere
        self.up_proj = nn.ModuleList([nn.Linear(self.latent_input_dim, self.latent_dim) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))])
        self.linear_proj1 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))]) 
        self.linear_proj2 = nn.ModuleList([nn.Linear(self.latent_dim, self.latent_dim) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))]) 
        self.down_proj = nn.ModuleList([nn.Linear(self.latent_dim, 1) for _ in range(len(self.time_scales_past) * len(self.time_scales_future))])
        
        self.pool_embs = nn.ModuleDict({
            str(scale): nn.AvgPool1d(kernel_size=scale*16)
            for scale in time_scales_past
        })

        # For joints and margs, the dominant dimension is the third one, so we pool over 128 times the future time scales
        self.pool_joints_margs = nn.ModuleDict({
            str(scale): nn.AvgPool1d(kernel_size=scale*128)
            for scale in time_scales_future
        })
        self.running_mean = RunningMean()
        

    def forward(self, audio_embs, audio_joints, audio_margs, mine_values):
        outputs = []
        
        # Downscale and encode audio_joints and audio_margs
        downscaled_joints = []
        downscaled_margs = []
        current_mine_count = mine_values[0]

        for j, scale in enumerate(self.time_scales_future):
            # Apply pooling to downscale, ensure the tensors are correctly shaped
            # Adjust transpose and view operations as necessary based on your actual data dimensions
            pooled_joint = self.pool_joints_margs[str(scale)](audio_joints[j].transpose(1, 2)).transpose(1, 2)
            pooled_marg = self.pool_joints_margs[str(scale)](audio_margs[j].transpose(1, 2)).transpose(1, 2)

            # Encode the downscaled tensors
            encoded_joint = self.audio_encoder(pooled_joint)
            encoded_marg = self.audio_encoder(pooled_marg)
            
            downscaled_joints.append(encoded_joint)
            downscaled_margs.append(encoded_marg)

        idx = 0
        for i, j in product(range(self.nr_past_timescales), range(self.nr_future_timescales)):
            s_past = self.time_scales_past[i]
            s_future = self.time_scales_future[j]

            # Downscale audio_embs using the appropriate pooling layer
            audio_emb_pooled = self.pool_embs[str(s_past)](audio_embs[i].transpose(1, 2)).transpose(1, 2)

            # Use the precomputed and encoded joints and margs
            z_joint, z_marg = downscaled_joints[j], downscaled_margs[j]

            # Concatenate and process with neural network layers
            # Ensure concatenation is along the correct dimension and results in correct shapes
            y_joint = torch.cat((audio_emb_pooled, z_joint), dim=1).view(1, -1)  # Adjust shapes as necessary
            y_marg = torch.cat((audio_emb_pooled, z_marg), dim=1).view(1, -1)  # Adjust shapes as necessary

            # Sequential operations for joint and marg
            y_joint = F.relu(self.up_proj[idx](y_joint))
            y_joint = F.relu(self.linear_proj1[idx](y_joint))
            y_joint = F.relu(self.linear_proj2[idx](y_joint))
            y_joint = F.relu(self.down_proj[idx](y_joint))
            
            y_marg = F.relu(self.up_proj[idx](y_marg))
            y_marg = F.relu(self.linear_proj1[idx](y_marg))
            y_marg = F.relu(self.linear_proj2[idx](y_marg))
            y_marg = F.relu(self.down_proj[idx](y_marg))
            
            outputs.append([y_joint, y_marg])
            idx += 1

        return outputs
    


class RewardModel():
    def __init__(self):
        # perform the multiscale MINE for input x_t (one audio buffer) at time step t, by calculating the mine score for each (i, j) pair, 
        # output the negative sum of the mine scores weighted by a learnt multiple of the avarage of feedback values 
        # across each respective future time_windows, for each respective back-timewindow, to be minimized by the optimizer
        pass

class GestureDataset(Dataset):  
    def __init__(self, gesture_dict):
        self.gestures = []
        self.labels = []
        self.label_mapping = {gesture: label for label, gesture in enumerate(gesture_dict.keys())}
        for gesture, gesture_list in gesture_dict.items():
            label = self.label_mapping[gesture]
            self.gestures.extend(gesture_list)
            self.labels.extend([label] * len(gesture_list))

    def __len__(self):
        return len(self.gestures)

    def __getitem__(self, idx):
        gesture = self.gestures[idx]
        label = self.labels[idx]
        return gesture, label

    def get_number_of_classes(self):
        return len(self.label_mapping)
    




class Trainer:
     
    def __init__(self, nr_classes):

        self.assets = {}
        self.prediction_classes = nr_classes
        self.model = SpiralnetRAVEClassifierGRU(self.prediction_classes)
        self.model.train()
        self.osc_input_port = 8004
        self.assets['model'] = self.model
        self.training_data_dict = {}
        #self.multiscale_mine = MultiscaleSequence_MINE()
        #self.reward_model = RewardModel()
        self.training_data_dict = {}
        


    def criterion(self, x, y):
        # calculate the mine score between the embedding vector (logits tensor) and the "next" audio input tensor
        # the model hears (or actually the downsampled pqmf of that audio buffer), over some future time window : 
        # mean(T(logits, future_window_joint)) - log(mean(exp(T(logits, future_window_marinal))))
        # + a nn.Linear projection of the input feedback from the RLHF app

        
        #Calculate : 
        # MINE(embedding[t - i], audio_input[t + j]) * w_{i, j} * sign(r_j) * exp(abs(r_j))
        # where w_{i, j} is the output of a neural network (nn.Linear X n) that is a function
        # of the embeddings and the rewards r_j, j = 1, 2, ...T_f, with the embeddings weighted
        # by a skewed normal to the left, and the rewards weighted by a skewed normal to the right
        # use s4 as MINE estimator over multiscale length sequences of pqmf embeddings and
        # latent embeddings

        pass

    def init_osc_server(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/training_data", self.handle_training_data)
        self.osc_ip = "0.0.0.0"  # Replace with your OSC server IP
        self.osc_port = self.osc_input_port       # Replace with your OSC server port
        self.osc_server = BlockingOSCUDPServer((self.osc_ip, self.osc_port), self.dispatcher)

    def start_osc_server(self):
        print("Starting OSC Server")
        self.osc_server.serve_forever()  # This will block until the server is stopped

    def stop_osc_server(self):
        if self.osc_server:
            self.osc_server.shutdown()
            print("OSC Server stopped")


    def handle_training_data(self, address, *args):
        # Quick unpacking assumes the structure and length of args is always the same and consistent
        pose_count, audio_buffer_count, logits_len, *remaining = args

        # Determine indices for slicing the remaining data
        logits_end = 3 + logits_len
        class_prediction_end = logits_end + 1
        softmax_values_end = class_prediction_end + remaining[class_prediction_end-3] + 1
        input_audio_end = softmax_values_end + remaining[softmax_values_end-3] + 1

        # Convert lists to tensors
        logits_tensor = torch.tensor(remaining[:logits_len])
        class_prediction_tensor = torch.tensor(remaining[logits_end:class_prediction_end])
        softmax_values_tensor = torch.tensor(remaining[class_prediction_end:softmax_values_end])
        input_audio_tensor = torch.tensor(remaining[softmax_values_end:input_audio_end]).reshape(1, 16, -1)

        # Efficiently get or create the list for audio_buffer_count
        tensors_list = self.training_data_dict.setdefault(audio_buffer_count, [])

        # Append the new data
        tensors_list.append([
            pose_count, logits_tensor, class_prediction_tensor,
            softmax_values_tensor, input_audio_tensor
        ])


    def handle_training_data_(self, address, *args):
        # Unpack arguments
        pose_count, audio_buffer_count = args[0], args[1]
        logits_len = args[2]
        logits_list = args[3:3 + logits_len]
        class_prediction_len = args[3 + logits_len]
        class_prediction_list = args[4 + logits_len:4 + logits_len + class_prediction_len]
        softmax_values_len = args[4 + logits_len + class_prediction_len]
        softmax_values_list = args[5 + logits_len + class_prediction_len:5 + logits_len + class_prediction_len + softmax_values_len]
        input_audio_len = args[5 + logits_len + class_prediction_len + softmax_values_len]
        input_audio_list = args[6 + logits_len + class_prediction_len + softmax_values_len:]

        # Convert lists to torch tensors
        logits_tensor = torch.tensor(logits_list)
        class_prediction_tensor = torch.tensor(class_prediction_list)
        softmax_values_tensor = torch.tensor(softmax_values_list)
        input_audio_tensor = torch.tensor(input_audio_list).reshape(1, 16, -1)

        # Store tensors in the dictionary
        if audio_buffer_count not in self.training_data_dict:
            self.training_data_dict[audio_buffer_count] = []
        self.training_data_dict[audio_buffer_count].append([
            pose_count, logits_tensor, class_prediction_tensor, 
            softmax_values_tensor, input_audio_tensor
        ])

    
    

    def compile_gestures_dict_and_initialise_model(self):
        self.store_as_torch_tensors(self.gestures)
        dataset = GestureDataset(self.gestures)
        nr_of_classes = dataset.get_number_of_classes()
        print('nr_of_classes = ', nr_of_classes)
        
        if self.current_nr_of_classes != nr_of_classes: # if 'model' not in self.assets:
            model = self.assets['model']
            optimizer = Adam(model.parameters(), lr=0.001)
            self.assets['model'] = model
            self.assets['optimizer'] = optimizer
            self.current_nr_of_classes = nr_of_classes
            
        else:
            model = self.assets['model']
            optimizer = self.assets['optimizer']
        
        return model, optimizer, dataset

    def train_model_RLHF(self):
            model = self.assets['model']
            dataset = self.assets['dataset']
            criterion = self.criterion
            optimizer = self.assets['optimizer']
                    
            num_epochs = 100
                    
            for epoch in range(num_epochs):
                running_loss = 0.0
                for i in range(len(dataset)):
                    gesture, label = dataset[i] # gesture, label, reward_label_prediction, reward_label_feedback = dataset[i]
                    gesture = gesture#.unsqueeze(0)  # Add batch dimension of 1
                    label = torch.tensor([label], dtype=torch.long)
                    
                    optimizer.zero_grad()
                    logits, prediction_label, value = model(gesture) # outputs, predicted_reward = model(gesture)
                    # predicted_reward = model2

                    ################################
                    # loss should be: 
                    # for each recorded label and for each recorded inference feedback for each label; if feedback reward label == 0: pass, if feedback reward label == 1; loss -= cross_entropy(exp(logits), logits) , if feedback reward label = -1 ; loss += cross_entropy(exp(logits), logits)
                    ################################

                    loss = criterion(logits.unsqueeze(0), torch.tensor([label], dtype=torch.long)) 
                    idx = int(prediction_label)
                    print(f"Idx: {idx}, Type: {type(self.reward_labels[idx])}")

                    for l in self.reward_labels[idx]:
                        
                        if l[0]==0: # the case of no or neutral reward received for the class label 
                            pass
                        elif l[0] < 0: # the case of negative reward received for the class label
                            loss += l[0] * torch.exp(l[1])
                        elif l[0] > 0: # the case of positive reward received the class label
                            loss += - l[0] * torch.exp(l[1])
                    
                    
                    loss.backward()
                    optimizer.step()
                    running_loss += loss.item() 
                    
                    # Print running loss
                    if i % 1 == 0:  # print every 10 samples
                        print(f'[Epoch {epoch + 1}, Sample {i + 1}] loss: {running_loss / 10:.3f}')
                        running_loss = 0.0

            print('Finished RLHF Training')
