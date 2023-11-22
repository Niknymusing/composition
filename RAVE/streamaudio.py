import torch
from pythonosc import dispatcher, osc_server
from collections import deque
import pyaudio
import numpy as np
from threading import Thread
import time
import wave

vae = torch.jit.load("/Users/nikny/Downloads/percussion.ts").eval()
pert = torch.linspace(-2, 2, 1)
output = []
SAMPLE_RATE = 44100
BUFFER_SIZE = 2048
p = pyaudio.PyAudio()
# Audio callback function
def audio_callback(in_data, frame_count, time_info, flag):
    # No gradients required for inference
    with torch.no_grad():
        # Convert the input buffer to a torch tensor
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        x = torch.tensor(audio_data).view(1, 1, -1)

        # Forward pass through the model
        z = vae.encode(x)
        z += pert# input_tensors.get(0)
        y = vae.decode(z)
        print(y)
        # Convert the output to bytes and return it
        out_data = y.numpy().astype(np.float32).tobytes()
        output.append(out_data)
        return (out_data, pyaudio.paContinue)
    
def audio_callback_(in_data, frame_count, time_info, flag):
    with torch.no_grad():
        #out.append(0)
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        x = torch.tensor(audio_data).view(1, 1, -1)

        # Forward pass through the model
        z = vae.encode(x)
        z += pert  # Assuming 'pert' is a tensor with the right shape to be added to 'z'
        y = vae.decode(z)
        #print(y.shape)

        # Check if y contains valid audio data

        # Convert the tensor to stereo if needed
        #y_stereo = y.repeat(1, 2, 1)  # Repeat the mono signal across two channels for stereo

        # Make sure the audio data is in the right range and type
        out_data = y.numpy().astype(np.float32).tobytes()
        output.append(out_data)

        return (out_data, pyaudio.paContinue)



stream = p.open(format=pyaudio.paFloat32,
                    channels=1,  # Use 1 channel for input
                    #output_channels=2,  # Use 2 channels for output if your device supports it
                    rate=SAMPLE_RATE,
                    input=True,
                    output=True,
                    input_device_index=1,
                    output_device_index=2,
                    frames_per_buffer=BUFFER_SIZE,
                    stream_callback=audio_callback_)


# Start the audio stream
stream.start_stream()

# Keep the main thread alive
try:
    while stream.is_active():
        # You could do some processing here, or just keep the thread alive
        time.sleep(0.1)
except KeyboardInterrupt:
    # Stop and close the stream and server
    stream.stop_stream()
    stream.close()
    p.terminate()

print("Shutting down")