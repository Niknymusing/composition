import torch
from torch import nn
import torch.nn.init as init
import torch.nn.functional as F
from collections import deque
from spiralnet import instantiate_model as instantiate_spiralnet 
from pqmf import CachedPQMF



class CircularBuffer:
    def __init__(self, size, tensor_shape):
        self.size = size
        self.buffer = deque(maxlen=size)
        self.tensor_shape = tensor_shape

    def add(self, tensor):
        
        self.buffer.append(tensor)

    def get(self, index):
        if index < 0 or index >= len(self.buffer):
            raise IndexError("Index out of range")
        return self.buffer[index]

class RNNCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional = True, batch_first = True, nonlinearity = 'relu'):
        super(RNNCell, self).__init__()
        
        # Adding Layer Normalization
        self.layer_norm = nn.LayerNorm(input_size)
        
        # instantiating RNN
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first, nonlinearity=nonlinearity)
        
        # Applying Xavier Uniform Initialization to RNN weights
        for param in self.rnn.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)

    def forward(self, x, state):
        x = self.layer_norm(x)
        output, hn = self.rnn(x, state)
        return output, hn
    

class GRUCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, nonlinearity='relu'):
        super(GRUCell, self).__init__()

        self.layer_norm = nn.LayerNorm(input_size)
        self.gru = nn.GRU(input_size, hidden_size, num_layers, bidirectional=bidirectional, batch_first=batch_first)
        self.gelu = nn.GELU()
        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)

    def forward(self, x, state = None):
        x = self.layer_norm(x)
        output, hn = self.gru(x, state)
        return self.gelu(output), hn
    

class TransformerCell(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, nonlinearity='tanh'):
        super(TransformerCell, self).__init__()

        self.layer_norm = nn.LayerNorm(input_size)
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_size, nhead=hidden_size // 30),
            num_layers=num_layers
        )

    def forward(self, x, state):
        x = self.layer_norm(x)
        output = self.transformer(x)
        return output, None  # Transformer does not maintain state like RNNs


class MineFF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, nonlinearity='relu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input, state):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output

####################################################################################################################

class Multiscale_MINE(nn.Module):


    def __init__(self, 
                 cell,
                 nr_time_scales = 8, 
                 nr_frequency_bands = 13, 
                 embedding_dim = 100, 
                 input_dim = 1024, 
                 nr_layers_per_timescale = 4, 
                 nr_spiralnet_layers = 16, 
                 delay_size = 3):
        
        super(Multiscale_MINE, self).__init__()  
        self.network_cell = GRUCell
        self.nr_time_scales = nr_time_scales
        self.nr_frequency_bands = nr_frequency_bands
        self.input_size = input_dim
        self.window_length = input_dim
        self.hop_size = self.window_length // 2 
        self.delay_size = delay_size
        self.nr_layers = nr_layers_per_timescale
        self.input_que = {}
        self.hidden_layers_dim = embedding_dim
        self.nr_spiralnet_layers = nr_spiralnet_layers
        
        self.input_counter = {0: 1}
        
        self.first_layers = nn.ModuleList(self.network_cell(input_size=self.input_size, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1))
        

        self.first_layers_ = nn.ModuleList(MineFF(input_size= self.input_size, 
                          hidden_size=self.hidden_layers_dim * 2, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'elu') for _ in range(self.nr_frequency_bands - 1))
        
        
        

        self.rnn_lists = nn.ModuleList(nn.ModuleList(self.network_cell(input_size= 4 * self.hidden_layers_dim, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1)) for _ in range(self.nr_time_scales)) 

        self.ffn_last = nn.ModuleList(nn.ModuleList(nn.Linear(2 * self.hidden_layers_dim, 1) for _ in range(self.nr_frequency_bands)) for _ in range(self.nr_time_scales))

        self.spiralnet = instantiate_spiralnet(nr_layers = self.nr_spiralnet_layers, output_dim = 2 * self.hidden_layers_dim)

        self.low_freq_hz = 0
        self.high_freq_hz = 44100 // 2  # Nyquist frequency for a 44.1kHz sample rate

        # Compute the Mel-spaced frequencies
        self.mel_band_limits = self.compute_mel_spaced_frequencies(self.low_freq_hz, self.high_freq_hz, self.nr_frequency_bands)
        self.fft_buffer = CircularBuffer(1, (1, self.hop_size))
        
        self.input_buffers = []
        for n in range(self.nr_frequency_bands):
            circular_buffer = CircularBuffer(self.delay_size, (1, self.input_size))
            self.input_buffers.append(circular_buffer)
        
        self.fft_buffers = [] 
        for n in range(self.nr_frequency_bands):
            fft_buffer = CircularBuffer(1, (1, self.input_size))
            fft_buffer.add(torch.zeros(1, self.window_length))
            self.fft_buffers.append(fft_buffer)
        
        self.output_buffers = []
        self.state_buffers = []
        for k in range(self.nr_time_scales):
            output_buffers_k = []
            state_buffers_k = []
            for n in range(self.nr_frequency_bands):
                output_buffer = CircularBuffer(self.delay_size, (1, 2 * self.hidden_layers_dim))
                state_buffer = CircularBuffer(self.delay_size, (2 * self.nr_layers, self.hidden_layers_dim))
                output_buffers_k.append(output_buffer)
                state_buffers_k.append(state_buffer)
            self.output_buffers.append(output_buffers_k)
            self.state_buffers.append(state_buffers_k)

        self.masks = []
        for i in range(len(self.mel_band_limits) - 1):
            # Convert mel_band_limits to integers
            start = int(self.mel_band_limits[i].item())
            end = int(self.mel_band_limits[i+1].item())

            # Create a mask with ones inside the frequency band and zeros outside
            mask = torch.zeros(1, self.window_length)
            mask[start:end] = 1.0

            # Consider the negative frequencies in a symmetric manner
            if end < self.window_length - start:
                mask[-end:-start] = 1.0
            self.masks.append(mask)

        self.analysis_window = torch.hamming_window(self.window_length)
        self.synthesis_window = torch.hamming_window(self.window_length)
        self.first_layer_norm = nn.LayerNorm(self.input_size)
        self.elu = nn.ELU()
        self.relu = nn.ReLU() 
        print('current version instantiated')

    def mine_score(self, transformed_joints, transformed_marginals):

        mine = self.relu(torch.mean(transformed_joints, 2) - torch.log(torch.mean(torch.exp(transformed_marginals), 2)))
    
        return -mine.sum(), mine

    def hz_to_mel(self, hz):
        """Convert hertz to mel scale."""
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(self, mel):
        """Convert mel scale to hertz."""
        return 700.0 * (10**(mel / 2595.0) - 1)

    def compute_mel_spaced_frequencies(self, low_freq_hz, high_freq_hz, num_points):
        """Generate Mel-spaced frequencies."""
        # Convert the frequency range to the Mel scale
        low_mel = self.hz_to_mel(torch.tensor(low_freq_hz))
        high_mel = self.hz_to_mel(torch.tensor(high_freq_hz))
        # Create an equally spaced grid in the Mel domain
        mel_points = torch.linspace(low_mel, high_mel, num_points)
        # Convert these Mel points back to the Hertz domain
        hz_points = self.mel_to_hz(mel_points)
        
        return hz_points
    
    def band_splitting(self, x, mel_band_limits): 
        # apply analysis_window
        # fft and masks, synthesis window
        # Store isolated bands
        x = self.analysis_window * x # zero-pad to the left by one self.hop_size?

        x_complex = torch.fft.fft(x)

        isolated_bands = []

        # Number of frequency bands
        nr_frequency_bands = len(mel_band_limits) - 1

        for i in range(nr_frequency_bands):

            band_isolated_complex = x_complex * self.masks[i]
            # Compute the inverse FFT to get the time-domain signal for this band
            curr_band = torch.fft.ifft(band_isolated_complex).real * self.synthesis_window # 
            #print('band_isolated.shape = ', band_isolated.shape)
            prev_band = self.fft_buffers[i].get(0) 
              #zeropad to the right by self.hop_size
            self.fft_buffers[i].add(curr_band)
            prev_band = F.pad(prev_band[:, -self.hop_size:], (0, self.hop_size))
            isolated_bands.append(prev_band + curr_band) 

        return isolated_bands
    
    def compute_statistics(self, x, poses):

        spiralnet_tensor = self.spiralnet(poses)

        output = []
        
        if self.input_counter[0] > self.delay_size:
            get_ind = self.delay_size - 1

        else:
            get_ind = self.input_counter[0] - 1

        curr_input = self.band_splitting(x, self.mel_band_limits)

        for n in range(self.nr_frequency_bands - 1):
            self.input_buffers[n].add(curr_input[n])

        # apply encodec to band splits, apply spiralnet encoder to mocap input
        # concat mocap embedding and audio embedding

        for n in range(self.nr_frequency_bands - 1):
            x = self.first_layer_norm(self.input_buffers[n].get(get_ind))
            #print('x.shape = ', x.shape)
            rnn_out, new_state = self.first_layers_[n](x, None), None
            #print('output first shape = ', rnn_out.shape)
            self.output_buffers[0][n].add(rnn_out)
            self.state_buffers[0][n].add(new_state)

        for k in range(self.nr_time_scales - 1):
            
            output_k = []
            for n in range(self.nr_frequency_bands - 1):
                # get input and state from the preceeding output_buffers
                rnn_input, state_input = torch.cat((self.output_buffers[k][n].get(get_ind), spiralnet_tensor), dim = 1),  self.state_buffers[k][n].get(get_ind) 
                rnn_out, new_state = self.rnn_lists[k][n](rnn_input, state_input)
                self.output_buffers[k+1][n].add(rnn_out)  # add the output of the rnn to its buffer
                self.state_buffers[k+1][n].add(new_state)
                out = self.output_buffers[k][n].get(get_ind)
                out = self.elu(self.ffn_last[k][n](out))
                output_k.append(out) # add elu nonlinearity
            
            output.append(torch.stack(output_k, dim=0).squeeze(2))
        
        self.input_counter[0] += 1
            
        return torch.stack(output, dim=0)
    


    
    def forward(self, audio_buffer, pose_joint, pose_marg):

        joint_statistics = self.compute_statistics(audio_buffer, pose_joint)
        marginal_statistics = self.compute_statistics(audio_buffer, pose_marg)
        mine_score, mine_matrix = self.mine_score(joint_statistics, marginal_statistics)

        return mine_score, mine_matrix
    



def mine_score(self, transformed_joints, transformed_marginals):

    mine = torch.mean(transformed_joints, 2) - torch.log(torch.mean(torch.exp(transformed_marginals), 2))
    
    return -mine.sum(), mine


####################################################################################################################


class Multiscale_MINE_PQMF(nn.Module):


    def __init__(self, 
                 cell,
                 nr_time_scales = 8, 
                 nr_frequency_bands = 16, 
                 embedding_dim = 100, 
                 input_dim = 1024, 
                 nr_layers_per_timescale = 4, 
                 nr_spiralnet_layers = 16, 
                 delay_size = 3):
        
        super(Multiscale_MINE, self).__init__()  
        self.network_cell = GRUCell
        self.nr_time_scales = nr_time_scales
        self.nr_frequency_bands = nr_frequency_bands
        self.input_size = input_dim
        self.window_length = input_dim
        self.hop_size = self.window_length // 2 
        self.delay_size = delay_size
        self.nr_layers = nr_layers_per_timescale
        self.input_que = {}
        self.hidden_layers_dim = embedding_dim
        self.nr_spiralnet_layers = nr_spiralnet_layers
        self.pqmf = CachedPQMF(attenuation = 82, n_band = self.nr_frequency_bands)
        
        self.input_counter = {0: 1}
        
        self.first_layers = nn.ModuleList(self.network_cell(input_size=self.input_size, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1))
        

        self.first_layers_ = nn.ModuleList(MineFF(input_size= self.input_size, 
                          hidden_size=self.hidden_layers_dim * 2, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'elu') for _ in range(self.nr_frequency_bands - 1))
        

        self.rnn_lists = nn.ModuleList(nn.ModuleList(self.network_cell(input_size= 4 * self.hidden_layers_dim, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1)) for _ in range(self.nr_time_scales)) 

        self.ffn_last = nn.ModuleList(nn.ModuleList(nn.Linear(2 * self.hidden_layers_dim, 1) for _ in range(self.nr_frequency_bands)) for _ in range(self.nr_time_scales))

        self.spiralnet = instantiate_spiralnet(nr_layers = self.nr_spiralnet_layers, output_dim = 2 * self.hidden_layers_dim)

    
        self.pqmf_buffer = CircularBuffer(1, (1, self.hop_size))
        
        self.input_buffers = []
        for n in range(self.nr_frequency_bands):
            circular_buffer = CircularBuffer(self.delay_size, (1, self.input_size))
            self.input_buffers.append(circular_buffer)
        
        self.pqmf_buffers = [] 
        for n in range(self.nr_frequency_bands):
            pqmf_buffer = CircularBuffer(1, (1, self.input_size))
            pqmf_buffer.add(torch.zeros(1, self.window_length))
            self.pqmf_buffers.append(pqmf_buffer)
        
        self.output_buffers = []
        self.state_buffers = []
        for k in range(self.nr_time_scales):
            output_buffers_k = []
            state_buffers_k = []
            for n in range(self.nr_frequency_bands):
                output_buffer = CircularBuffer(self.delay_size, (1, 2 * self.hidden_layers_dim))
                state_buffer = CircularBuffer(self.delay_size, (2 * self.nr_layers, self.hidden_layers_dim))
                output_buffers_k.append(output_buffer)
                state_buffers_k.append(state_buffer)
            self.output_buffers.append(output_buffers_k)
            self.state_buffers.append(state_buffers_k)

        self.first_layer_norm = nn.LayerNorm(self.input_size)
        self.elu = nn.ELU()
        self.relu = nn.ReLU() 
        print('multiscale mine with PQMF band splitting instantiated')

    def mine_score(self, transformed_joints, transformed_marginals):

        mine = self.relu(torch.mean(transformed_joints, 2) - torch.log(torch.mean(torch.exp(transformed_marginals), 2)))
    
        return -mine.sum(), mine
    
    def band_splitting(self, x): 

        x = self.analysis_window * x # zero-pad to the left by one self.hop_size?

        x_complex = torch.fft.fft(x)

        isolated_bands = []

        for i in range(self.nr_frequency_bands):

            band_isolated_complex = x_complex * self.masks[i]
            # Compute the inverse FFT to get the time-domain signal for this band
            curr_band = torch.fft.ifft(band_isolated_complex).real * self.synthesis_window # 
            #print('band_isolated.shape = ', band_isolated.shape)
            prev_band = self.fft_buffers[i].get(0) 
              #zeropad to the right by self.hop_size
            self.fft_buffers[i].add(curr_band)
            prev_band = F.pad(prev_band[:, -self.hop_size:], (0, self.hop_size))
            isolated_bands.append(prev_band + curr_band) 

        return isolated_bands
    
    def compute_statistics(self, x, poses):

        spiralnet_tensor = self.spiralnet(poses)

        output = []
        
        if self.input_counter[0] > self.delay_size:
            get_ind = self.delay_size - 1

        else:
            get_ind = self.input_counter[0] - 1

        curr_input = self.band_splitting(x, self.mel_band_limits)

        for n in range(self.nr_frequency_bands - 1):
            self.input_buffers[n].add(curr_input[n])

        # apply encodec to band splits, apply spiralnet encoder to mocap input
        # concat mocap embedding and audio embedding

        for n in range(self.nr_frequency_bands - 1):
            x = self.first_layer_norm(self.input_buffers[n].get(get_ind))
            #print('x.shape = ', x.shape)
            rnn_out, new_state = self.first_layers_[n](x, None), None
            #print('output first shape = ', rnn_out.shape)
            self.output_buffers[0][n].add(rnn_out)
            self.state_buffers[0][n].add(new_state)

        for k in range(self.nr_time_scales - 1):
            
            output_k = []
            for n in range(self.nr_frequency_bands - 1):
                # get input and state from the preceeding output_buffers
                rnn_input, state_input = torch.cat((self.output_buffers[k][n].get(get_ind), spiralnet_tensor), dim = 1),  self.state_buffers[k][n].get(get_ind) 
                rnn_out, new_state = self.rnn_lists[k][n](rnn_input, state_input)
                self.output_buffers[k+1][n].add(rnn_out)  # add the output of the rnn to its buffer
                self.state_buffers[k+1][n].add(new_state)
                out = self.output_buffers[k][n].get(get_ind)
                out = self.elu(self.ffn_last[k][n](out))
                output_k.append(out) # add elu nonlinearity
            
            output.append(torch.stack(output_k, dim=0).squeeze(2))
        
        self.input_counter[0] += 1
            
        return torch.stack(output, dim=0)
    
    
    def forward(self, audio_buffer, pose_joint, pose_marg):

        joint_statistics = self.compute_statistics(audio_buffer, pose_joint)
        marginal_statistics = self.compute_statistics(audio_buffer, pose_marg)
        mine_score, mine_matrix = self.mine_score(joint_statistics, marginal_statistics)

        return mine_score, mine_matrix
    






####################################################################################################################



class Multiscale_MINE_oldV(nn.Module):

    def __init__(self, nr_time_scales = 8, 
                 nr_frequency_bands = 13, 
                 embedding_dim = 100, 
                 input_dim = 1024, 
                 nr_layers_per_timescale = 4, 
                 nr_spiralnet_layers = 16, 
                 delay_size = 3):
        
        super(Multiscale_MINE_oldV, self).__init__()  
        
        self.nr_time_scales = nr_time_scales
        self.nr_frequency_bands = nr_frequency_bands
        self.input_size = input_dim
        self.window_length = input_dim
        self.hop_size = self.window_length // 2 
        self.delay_size = delay_size
        self.nr_layers = nr_layers_per_timescale
        self.input_que = {}
        self.hidden_layers_dim = embedding_dim
        self.nr_spiralnet_layers = nr_spiralnet_layers
        
        self.input_counter = {0: 1}
        
        self.rnn_first = nn.ModuleList(nn.RNN(input_size=self.input_size, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1))

        self.rnn_lists = nn.ModuleList(nn.ModuleList(RNNCell(input_size= 4 * self.hidden_layers_dim, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1)) for _ in range(self.nr_time_scales)) 

        self.ffn_last = nn.ModuleList(nn.ModuleList(nn.Linear(2 * self.hidden_layers_dim, 1) for _ in range(self.nr_frequency_bands)) for _ in range(self.nr_time_scales))

        self.spiralnet = instantiate_spiralnet(nr_layers = self.nr_spiralnet_layers, output_dim = 2 * self.hidden_layers_dim)

        self.low_freq_hz = 0
        self.high_freq_hz = 44100 // 2  # Nyquist frequency for a 44.1kHz sample rate

        # Compute the Mel-spaced frequencies
        self.mel_band_limits = self.compute_mel_spaced_frequencies(self.low_freq_hz, self.high_freq_hz, self.nr_frequency_bands)
        self.fft_buffer = CircularBuffer(1, (1, self.hop_size))
        
        self.input_buffers = []
        for n in range(self.nr_frequency_bands):
            circular_buffer = CircularBuffer(self.delay_size, (1, self.input_size))
            self.input_buffers.append(circular_buffer)
        
        self.fft_buffers = [] 
        for n in range(self.nr_frequency_bands):
            fft_buffer = CircularBuffer(1, (1, self.input_size))
            fft_buffer.add(torch.zeros(1, self.window_length))
            self.fft_buffers.append(fft_buffer)
        
        self.output_buffers = []
        self.state_buffers = []
        for k in range(self.nr_time_scales):
            output_buffers_k = []
            state_buffers_k = []
            for n in range(self.nr_frequency_bands):
                output_buffer = CircularBuffer(self.delay_size, (1, 2 * self.hidden_layers_dim))
                state_buffer = CircularBuffer(self.delay_size, (2 * self.nr_layers, self.hidden_layers_dim))
                output_buffers_k.append(output_buffer)
                state_buffers_k.append(state_buffer)
            self.output_buffers.append(output_buffers_k)
            self.state_buffers.append(state_buffers_k)

        self.masks = []
        for i in range(len(self.mel_band_limits) - 1):
            # Convert mel_band_limits to integers
            start = int(self.mel_band_limits[i].item())
            end = int(self.mel_band_limits[i+1].item())

            # Create a mask with ones inside the frequency band and zeros outside
            mask = torch.zeros(1, self.window_length)
            mask[start:end] = 1.0

            # Consider the negative frequencies in a symmetric manner
            if end < self.window_length - start:
                mask[-end:-start] = 1.0
            self.masks.append(mask)

        self.analysis_window = torch.hamming_window(self.window_length)
        self.synthesis_window = torch.hamming_window(self.window_length)
        self.first_layer_norm = nn.LayerNorm(self.input_size)
        self.elu = nn.ELU()

    def hz_to_mel(self, hz):
        """Convert hertz to mel scale."""
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(self, mel):
        """Convert mel scale to hertz."""
        return 700.0 * (10**(mel / 2595.0) - 1)

    def compute_mel_spaced_frequencies(self, low_freq_hz, high_freq_hz, num_points):
        """Generate Mel-spaced frequencies."""
        # Convert the frequency range to the Mel scale
        low_mel = self.hz_to_mel(torch.tensor(low_freq_hz))
        high_mel = self.hz_to_mel(torch.tensor(high_freq_hz))
        
        # Create an equally spaced grid in the Mel domain
        mel_points = torch.linspace(low_mel, high_mel, num_points)
        
        # Convert these Mel points back to the Hertz domain
        hz_points = self.mel_to_hz(mel_points)
        
        return hz_points
    
    def band_splitting(self, x, mel_band_limits):
        # 
        # apply analysis_window
        # fft and masks, synthesis window
        # Store isolated bands
        x = self.analysis_window * x # zero-pad to the left by one self.hop_size?

        x_complex = torch.fft.fft(x)

        isolated_bands = []

        # Number of frequency bands
        nr_frequency_bands = len(mel_band_limits) - 1

        for i in range(nr_frequency_bands):

            band_isolated_complex = x_complex * self.masks[i]
            # Compute the inverse FFT to get the time-domain signal for this band
            curr_band = torch.fft.ifft(band_isolated_complex).real * self.synthesis_window # 
            #print('band_isolated.shape = ', band_isolated.shape)
            prev_band = self.fft_buffers[i].get(0) 
              #zeropad to the right by self.hop_size
            self.fft_buffers[i].add(curr_band)
            prev_band = F.pad(prev_band[:, -self.hop_size:], (0, self.hop_size))
            isolated_bands.append(prev_band + curr_band) #circular_buffers[i].add(band_isolated) #

        return isolated_bands
    
    def compute_statistics(self, x, poses):

        spiralnet_tensor = self.spiralnet(poses)

        output = []
        
        if self.input_counter[0] > self.delay_size:
            get_ind = self.delay_size - 1
            #print(' in the case self.input_counter[0] - 1 > self.delay_size, get_ind = ', get_ind)
        else:
            get_ind = self.input_counter[0] - 1
            #print('global get_ind = ', get_ind)


        curr_input = self.band_splitting(x, self.mel_band_limits)

        for n in range(self.nr_frequency_bands - 1):
            self.input_buffers[n].add(curr_input[n])

        # apply encodec to band splits, apply spiralnet encoder to mocap input
        # concat mocap embedding and audio embedding

        for n in range(self.nr_frequency_bands - 1):
            x = self.first_layer_norm(self.input_buffers[n].get(get_ind))
            rnn_out, new_state = self.rnn_first[n](x)
            self.output_buffers[0][n].add(rnn_out)
            self.state_buffers[0][n].add(new_state)

        for k in range(self.nr_time_scales - 1):
            
            output_k = []
            for n in range(self.nr_frequency_bands - 1):
                #print('n = ', n, ' ' 'k = ', k , ' get_ind = ', get_ind)
                rnn_input, state_input = torch.cat((self.output_buffers[k][n].get(get_ind), spiralnet_tensor), dim = 1),  self.state_buffers[k][n].get(get_ind) # get input and state from the preceeding output_buffers
                #print('rnn_input.shape = ', rnn_input.shape)
                rnn_out, new_state = self.rnn_lists[k][n](rnn_input, state_input)
                self.output_buffers[k+1][n].add(rnn_out)  # add the output of the rnn to its buffer
                self.state_buffers[k+1][n].add(new_state)
                out = self.output_buffers[k][n].get(get_ind)
                out = self.elu(self.ffn_last[k][n](out))
                #print('self.ffn_last[k][n](out).shape = ', self.ffn_last[k][n](out).shape)
                output_k.append(out) # add elu nonlinearity
            #print('torch.stack(output_k, dim=0).shape = ', torch.stack(output_k, dim=0).shape) # (12, 32, 1)
            output.append(torch.stack(output_k, dim=0).squeeze(2))
        
        self.input_counter[0] += 1
            
        return torch.stack(output, dim=0)
    
    #def mine_score(self, joints, marginals):

    #    for k in range(self.nr_time_scales):
    #        for n in range(self.nr_frequency_bands):

    
    def forward(self, audio_buffer, pose_joint, pose_marg):

        joint_statistics = self.compute_statistics(audio_buffer, pose_joint)
        marginal_statistics = self.compute_statistics(audio_buffer, pose_marg)
        mine_matrix = torch.mean(joint_statistics, 2) - torch.log(torch.mean(torch.exp(marginal_statistics), 2))

        return -mine_matrix.sum(), mine_matrix






####################################################################################################################

class MineFF(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bidirectional=True, batch_first=True, nonlinearity='elu'):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        nn.init.normal_(self.fc1.weight,std=0.02)
        nn.init.constant_(self.fc1.bias, 0)
        nn.init.normal_(self.fc2.weight,std=0.02)
        nn.init.constant_(self.fc2.bias, 0)
        nn.init.normal_(self.fc3.weight,std=0.02)
        nn.init.constant_(self.fc3.bias, 0)
        
    def forward(self, input, state):
        output = F.elu(self.fc1(input))
        output = F.elu(self.fc2(output))
        output = self.fc3(output)
        return output




class Multiscale_MINE_test(nn.Module):


    def __init__(self, 
                 cell,
                 nr_time_scales = 8, 
                 nr_frequency_bands = 13, 
                 embedding_dim = 100, 
                 input_dim = 1024, 
                 nr_layers_per_timescale = 4, 
                 nr_spiralnet_layers = 16, 
                 delay_size = 3):
        
        super(Multiscale_MINE_test, self).__init__()  
        self.network_cell = GRUCell
        self.nr_time_scales = nr_time_scales
        self.nr_frequency_bands = nr_frequency_bands
        self.input_size = input_dim
        self.window_length = input_dim
        self.hop_size = self.window_length // 2 
        self.delay_size = delay_size
        self.nr_layers = nr_layers_per_timescale
        self.input_que = {}
        self.hidden_layers_dim = embedding_dim
        self.nr_spiralnet_layers = nr_spiralnet_layers
        
        self.input_counter = {0: 1}
        # todos: 
        # msybe add layer norms, and maybe dropout
        # init weights
        self.first_layers = nn.ModuleList(self.network_cell(input_size=self.input_size, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1))
        

        self.first_layers_ = nn.ModuleList(MineFF(input_size= self.input_size, 
                          hidden_size=self.hidden_layers_dim * 2, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'elu') for _ in range(self.nr_frequency_bands - 1))
        
        
        

        self.rnn_lists = nn.ModuleList(nn.ModuleList(self.network_cell(input_size= 4 * self.hidden_layers_dim, 
                          hidden_size=self.hidden_layers_dim, 
                          num_layers=self.nr_layers, 
                          bidirectional=True, 
                          batch_first=True,
                          nonlinearity = 'relu') for _ in range(self.nr_frequency_bands - 1)) for _ in range(self.nr_time_scales)) 

        self.ffn_last = nn.ModuleList(nn.ModuleList(nn.Linear(2 * self.hidden_layers_dim, 1) for _ in range(self.nr_frequency_bands)) for _ in range(self.nr_time_scales))

        self.spiralnet = instantiate_spiralnet(nr_layers = self.nr_spiralnet_layers, output_dim = 2 * self.hidden_layers_dim)

        self.low_freq_hz = 0
        self.high_freq_hz = 44100 // 2  # Nyquist frequency for a 44.1kHz sample rate

        # Compute the Mel-spaced frequencies
        self.mel_band_limits = self.compute_mel_spaced_frequencies(self.low_freq_hz, self.high_freq_hz, self.nr_frequency_bands)
        self.fft_buffer = CircularBuffer(1, (1, self.hop_size))
        
        self.input_buffers = []
        for n in range(self.nr_frequency_bands):
            circular_buffer = CircularBuffer(self.delay_size, (1, self.input_size))
            self.input_buffers.append(circular_buffer)
        
        self.fft_buffers = [] 
        for n in range(self.nr_frequency_bands):
            fft_buffer = CircularBuffer(1, (1, self.input_size))
            fft_buffer.add(torch.zeros(1, self.window_length))
            self.fft_buffers.append(fft_buffer)
        
        self.output_buffers = []
        self.state_buffers = []
        for k in range(self.nr_time_scales):
            output_buffers_k = []
            state_buffers_k = []
            for n in range(self.nr_frequency_bands):
                output_buffer = CircularBuffer(self.delay_size, (1, 2 * self.hidden_layers_dim))
                state_buffer = CircularBuffer(self.delay_size, (2 * self.nr_layers, self.hidden_layers_dim))
                output_buffers_k.append(output_buffer)
                state_buffers_k.append(state_buffer)
            self.output_buffers.append(output_buffers_k)
            self.state_buffers.append(state_buffers_k)

        self.masks = []
        for i in range(len(self.mel_band_limits) - 1):
            # Convert mel_band_limits to integers
            start = int(self.mel_band_limits[i].item())
            end = int(self.mel_band_limits[i+1].item())

            # Create a mask with ones inside the frequency band and zeros outside
            mask = torch.zeros(1, self.window_length)
            mask[start:end] = 1.0

            # Consider the negative frequencies in a symmetric manner
            if end < self.window_length - start:
                mask[-end:-start] = 1.0
            self.masks.append(mask)

        self.analysis_window = torch.hamming_window(self.window_length)
        self.synthesis_window = torch.hamming_window(self.window_length)
        self.first_layer_norm = nn.LayerNorm(self.input_size)
        self.elu = nn.ELU()

    def mine_score(self, transformed_joints, transformed_marginals):

        mine = torch.mean(transformed_joints, 2) - torch.log(torch.mean(torch.exp(transformed_marginals), 2))
    
        return -mine.sum(), mine

    def hz_to_mel(self, hz):
        """Convert hertz to mel scale."""
        return 2595.0 * torch.log10(1.0 + hz / 700.0)

    def mel_to_hz(self, mel):
        """Convert mel scale to hertz."""
        return 700.0 * (10**(mel / 2595.0) - 1)

    def compute_mel_spaced_frequencies(self, low_freq_hz, high_freq_hz, num_points):
        """Generate Mel-spaced frequencies."""
        # Convert the frequency range to the Mel scale
        low_mel = self.hz_to_mel(torch.tensor(low_freq_hz))
        high_mel = self.hz_to_mel(torch.tensor(high_freq_hz))
        
        # Create an equally spaced grid in the Mel domain
        mel_points = torch.linspace(low_mel, high_mel, num_points)
        
        # Convert these Mel points back to the Hertz domain
        hz_points = self.mel_to_hz(mel_points)
        
        return hz_points
    
    def band_splitting(self, x, mel_band_limits):
        # 
        # apply analysis_window
        # fft and masks, synthesis window
        # Store isolated bands
        x = self.analysis_window * x # zero-pad to the left by one self.hop_size?

        x_complex = torch.fft.fft(x)

        isolated_bands = []

        # Number of frequency bands
        nr_frequency_bands = len(mel_band_limits) - 1

        for i in range(nr_frequency_bands):

            band_isolated_complex = x_complex * self.masks[i]
            # Compute the inverse FFT to get the time-domain signal for this band
            curr_band = torch.fft.ifft(band_isolated_complex).real * self.synthesis_window # 
            #print('band_isolated.shape = ', band_isolated.shape)
            prev_band = self.fft_buffers[i].get(0) 
              #zeropad to the right by self.hop_size
            self.fft_buffers[i].add(curr_band)
            prev_band = F.pad(prev_band[:, -self.hop_size:], (0, self.hop_size))
            isolated_bands.append(prev_band + curr_band) #circular_buffers[i].add(band_isolated) #

        return isolated_bands
    
    def compute_statistics(self, x, poses):

        spiralnet_tensor = self.spiralnet(poses)
        #print('spiralnet_tensor.shape = ', spiralnet_tensor.shape)

        output = []
        
        if self.input_counter[0] > self.delay_size:
            get_ind = self.delay_size - 1
            #print(' in the case self.input_counter[0] - 1 > self.delay_size, get_ind = ', get_ind)
        else:
            get_ind = self.input_counter[0] - 1
            #print('global get_ind = ', get_ind)


        curr_input = self.band_splitting(x, self.mel_band_limits)

        for n in range(self.nr_frequency_bands - 1):
            self.input_buffers[n].add(curr_input[n])

        # apply encodec to band splits, apply spiralnet encoder to mocap input
        # concat mocap embedding and audio embedding

        for n in range(self.nr_frequency_bands - 1):
            x = self.first_layer_norm(self.input_buffers[n].get(get_ind))
            #print('x.shape = ', x.shape)
            rnn_out, new_state = self.first_layers_[n](x, None), None
            #print('output first shape = ', rnn_out.shape)
            self.output_buffers[0][n].add(rnn_out)
            self.state_buffers[0][n].add(new_state)

        for k in range(self.nr_time_scales - 1):
            
            output_k = []
            for n in range(self.nr_frequency_bands - 1):
                #print('n = ', n, ' ' 'k = ', k , ' get_ind = ', get_ind)
                rnn_input, state_input = torch.cat((self.output_buffers[k][n].get(get_ind), spiralnet_tensor), dim = 1),  self.state_buffers[k][n].get(get_ind) # get input and state from the preceeding output_buffers
                #print('rnn_input.shape = ', rnn_input.shape)
                rnn_out, new_state = self.rnn_lists[k][n](rnn_input, state_input)
                self.output_buffers[k+1][n].add(rnn_out)  # add the output of the rnn to its buffer
                self.state_buffers[k+1][n].add(new_state)
                out = self.output_buffers[k][n].get(get_ind)
                out = self.elu(self.ffn_last[k][n](out))
                #print('self.ffn_last[k][n](out).shape = ', self.ffn_last[k][n](out).shape)
                output_k.append(out) # add elu nonlinearity
            #print('torch.stack(output_k, dim=0).shape = ', torch.stack(output_k, dim=0).shape) # (12, 32, 1)
            output.append(torch.stack(output_k, dim=0).squeeze(2))
        
        self.input_counter[0] += 1
            
        return torch.stack(output, dim=0)
    
    #def mine_score(self, joints, marginals):

    #    for k in range(self.nr_time_scales):
    #        for n in range(self.nr_frequency_bands):

    
    def forward(self, audio_buffer, pose_joint, pose_marg):

        joint_statistics = self.compute_statistics(audio_buffer, pose_joint)
        marginal_statistics = self.compute_statistics(audio_buffer, pose_marg)
        mine_score, mine_matrix = self.mine_score(joint_statistics, marginal_statistics)

        return mine_score, mine_matrix
    



def mine_score(self, transformed_joints, transformed_marginals):

    mine = torch.mean(transformed_joints, 2) - torch.log(torch.mean(torch.exp(transformed_marginals), 2))
    
    return -mine.sum(), mine
