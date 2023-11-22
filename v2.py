import soundcard as sc
import numpy as np
import wave
import time
import torch
import librosa as li 
import soundfile as sf
from typing import Any, Callable, Optional, Union
import torch.nn.functional as F
from einops import repeat
from torch import nn
import time
import pyaudio
import wave
import numpy as np
import soundfile as sf
import torch
from pythonosc import dispatcher, osc_server
from collections import deque
import pyaudio
import numpy as np
from threading import Thread
import argparse
import os
from collections import deque
import queue
import threading

# Define the default paths
DEFAULT_WAV_PATH = "/Users/nikny/musing_instruments/data/stravinski_wav/01 Petroushka (Original 1911 Version), First Scene_ I. The Shrove-tide Fair.wav"
DEFAULT_MODEL_PATHS = {
    1: "/Users/nikny/Downloads/percussion.ts",
    2: "/Users/nikny/Downloads/nasa.ts",
    3: "/Users/nikny/Downloads/vintage.ts",  # Too heavy
    4: "/Users/nikny/Downloads/brasileiro0.ts", 
    5: "/Users/nikny/Downloads/voice_hifitts_b2048_r48000_z16.ts",
    6: "/Users/nikny/Downloads/voice_jvs_b2048_r44100_z16.ts",
    7: "/Users/nikny/Downloads/voice_vctk_b2048_r44100_z22.ts",
    8: "/Users/nikny/Downloads/voice_vocalset_b2048_r48000_z16.ts",
    9: "/Users/nikny/Downloads/water_pondbrain_b2048_r48000_z16.ts",
    10: "/Users/nikny/Downloads/magnets_b2048_r48000_z8.ts",
    11: "/Users/nikny/Downloads/marinemammals_pondbrain_b2048_r48000_z20.ts",
    12: "/Users/nikny/Downloads/humpbacks_pondbrain_b2048_r48000_z20.ts",
    13: "/Users/nikny/Downloads/organ_archive_b2048_r48000_z16.ts",
    14: "/Users/nikny/Downloads/organ_bach_b2048_r48000_z16.ts",
    15: "/Users/nikny/Downloads/sax_soprano_franziskaschroeder_b2048_r48000_z20.ts",
    16: "/Users/nikny/Downloads/guitar_iil_b2048_r48000_z16.ts",
    17: "/Users/nikny/Downloads/birds_motherbird_b2048_r48000_z16.ts",
    18: "/Users/nikny/Downloads/birds_pluma_b2048_r48000_z12.ts",
    19: "/Users/nikny/Downloads/mrp_strengjavera_b2048_r44100_z16.ts"
}

# Set up argparse to handle command line arguments
parser = argparse.ArgumentParser(description="Run the audio processing script.")
parser.add_argument("--port", type=int, default=4590, help="port to receive osc messages for latent perturbations")
parser.add_argument('--model', type=int, help="Model path or number (1-19)", default=1)
parser.add_argument('--wav', type=str, help="Custom .wav file path", default=DEFAULT_WAV_PATH)
parser.add_argument('--rec', action='store_true', help="Record audio from input device")
parser.add_argument('--buffer_size', type=int, default=4096, help="set buffer_size for the model processing")
parser.add_argument('--model_lookback', type=int, default=16000, help="the size of the model's accessed audio context window")
args = parser.parse_args()

osc_port = args.port
buffer_length = args.buffer_size #8192
model_lookback = args.model_lookback

# Function to monitor for Enter key press
def keypress_monitor(stop_recording_flag):
    input()
    stop_recording_flag.put(True)

# Recording function 
def record_audio(output_filename, warmup_time = 0.5):
    chunk = 1024
    format = pyaudio.paInt16
    channels = 1
    rate = 44100
    record_seconds = 5
    warmup_chunks = int(rate / chunk * warmup_time)

    p = pyaudio.PyAudio()

    stream = p.open(format=format, channels=channels, rate=rate, input=True, frames_per_buffer=chunk)

    print("Recording for 5 seconds...")

    frames = []

    for i in range(0, int(rate / chunk * record_seconds)):
        data = stream.read(chunk)
        if i >= warmup_chunks:
            frames.append(data)

    print("Finished recording.")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(output_filename, 'wb')
    wf.setnchannels(channels)
    wf.setsampwidth(p.get_sample_size(format))
    wf.setframerate(rate)
    wf.writeframes(b''.join(frames))
    wf.close()


# If recording flag is present, record and set the wav_path to the recorded file
if args.rec:
    recorded_wav_path = "recorded_audio.wav"
    print('starting rec into model ', DEFAULT_MODEL_PATHS[args.model])
    record_audio(recorded_wav_path)
    wav_path = recorded_wav_path
else:
    # Use the provided wav file or the default
    wav_path = args.wav if args.wav and os.path.isfile(args.wav) else DEFAULT_WAV_PATH

# Determine the model path
if args.model in DEFAULT_MODEL_PATHS:
    model_path = DEFAULT_MODEL_PATHS[args.model]
elif os.path.exists(args.model):
    model_path = args.model
else:
    print(f"Model path provided is not valid: {args.model}. Using default model 1.")
    model_path = DEFAULT_MODEL_PATHS['1']

# Determine the .wav file path
#wav_path = args.wav if os.path.exists(args.wav) else DEFAULT_WAV_PATH
if wav_path == DEFAULT_WAV_PATH and args.wav != DEFAULT_WAV_PATH:
    wav_path = DEFAULT_WAV_PATH
    print(f"Custom .wav path provided is not valid: {args.wav}. Using default wav path.")


def convert_wav_to_float32(input_path, output_path):
    # Open the wav file
    with wave.open(input_path, 'rb') as wav_file:
        # Extract audio data and parameters
        n_channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        framerate = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)
        # Convert audio data to numpy array depending on the sample width
        if sample_width == 1:  # 8-bit WAV files are unsigned
            data = np.frombuffer(audio_data, dtype=np.uint8) - 128
        elif sample_width == 2:  # 16-bit WAV files are signed
            data = np.frombuffer(audio_data, dtype=np.int16)
        else:
            raise ValueError("Only supports 8 or 16 bit audio formats.")
        # Normalize the data to the range between -1.0 and 1.0
        max_int_value = float(2 ** (8 * sample_width - 1))
        data = data / max_int_value
        # Write the data to a new file
        sf.write(output_path, data, framerate, 'FLOAT')

def generate_perturbation_tensors(audio, model):
    tensor = model.encode(audio[:, :, 0:model_lookback])
    init_tensor = torch.ones_like(tensor)
    perturbation_tensors = {0: init_tensor}
    return init_tensor#perturbation_tensors

convert_wav_to_float32(wav_path, 'output_path.wav')
torch.set_grad_enabled(False)
model = torch.jit.load(model_path).eval()
if args.rec:
    x = li.load('output_path.wav')[0]
    print('loaded recorded_audio.wav')
else:
    x = li.load("output_path.wav")[0]
x = torch.from_numpy(x).reshape(1, 1, -1)
x = x[:,:,x.shape[2]%buffer_length:] # cut of the x.shape[0]%buffer_length number of last samples of the tensor, to not break the loop over buffers below
perturbation_tensor = generate_perturbation_tensors(x, model)

perturbation_idx = {0: 0}

def osc_callback0(addr, *args):

    perturbation_tensor[0][0] *= args[0]#tensor
    perturbation_idx[0]=0
    print('updated 0 ', args[0])

def osc_callback1(addr, *args):

    perturbation_tensor[0][1] *= args[0]#tensor
    perturbation_idx[0]=1
    print('updated 1 ', args[0])

def osc_callback2(addr, *args):

    perturbation_tensor[0][2] *= args[0]#tensor
    perturbation_idx[0]=2
    print('updated 2 ', args[0])

def osc_callback3(addr, *args):

    perturbation_tensor[0][3] *= args[0]#tensor
    perturbation_idx[0]=3
    print('updated 3 ', args[0])

# Setting up the OSC server
def start_osc_server(ip, port):
    disp = dispatcher.Dispatcher()
    disp.map("/latent_perturbations0", osc_callback0)
    disp.map("/latent_perturbations1", osc_callback1)
    disp.map("/latent_perturbations2", osc_callback2)
    disp.map("/latent_perturbations3", osc_callback3)
    server = osc_server.ThreadingOSCUDPServer((ip, port), disp)
    print(f"Serving on {server.server_address}")
    server.serve_forever()




# Replace 'input_path.wav' and 'output_path.wav' with the actual paths




p = pyaudio.PyAudio()
osc_thread = Thread(target=start_osc_server, args=('127.0.0.1', osc_port))
osc_thread.start()
# Assuming the audio is mono (1 channel) and float32 format, adjust as needed
stream = p.open(format=pyaudio.paFloat32,  # float32 format
                channels=1,                # Mono audio
                rate=44100, 
                frames_per_buffer=1024,               # Sample rate
                output=True,               # Output stream
                output_device_index=2) 


prev_buffer = {0:torch.zeros(2*buffer_length)}

#load audio_tensors
#configure osc callback to receive updated index list audio_tensors_idx

x = x[:, :, x.shape[2]%buffer_length:]

for _ in iter(int, 1): 
    t = time.time()
    for i in range((x.shape[2] // buffer_length)-13):
            #t = time.time()
        z = model.encode(x[:, :, (i*buffer_length):(model_lookback + i*buffer_length)])     #(model_lookback + i*buffer_length):(2*model_lookback + i*buffer_length)])
        z[:, perturbation_idx[0]] += 0.1*perturbation_tensor[0][perturbation_idx[0]]        #perturbation_tensors[0]# torch.linspace(-2, 2, z.shape[-1])
        y = model.decode(z).numpy().reshape(-1)
        last_samples = y[-buffer_length:].astype('float32').tobytes()
        # Write to PyAudio stream
        stream.write(last_samples)
        #print(y.shape)
        #print(time.time() - t)
    print('one play-through, time: ', time.time()-t)

stream.stop_stream()
stream.close()

# Terminate PyAudio
p.terminate()