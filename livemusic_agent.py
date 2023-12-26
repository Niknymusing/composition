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
import torch.nn.functional as F
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

osc_output_ip = '127.0.0.1'
osc_output_port = 11000
osc_input_ip = '127.0.0.1'
osc_input_port = 4592

KERNEL_SIZE = 3
DILATIONS = [
    [1, 3, 9],
    [1, 3, 9],
    [1, 3, 9],
    [1, 3],
]
RATIOS = [4, 4, 4, 2]
CAPACITY = 64#96#64#
NOISE_AUGMENTATION = 0
LATENT_SIZE = 16
N_BAND = LATENT_SIZE# 16

pqmf = CachedPQMF(n_band = N_BAND, attenuation = 100)
encoder = EncoderV2(data_size = N_BAND, capacity = CAPACITY, ratios = RATIOS, 
                    latent_size = LATENT_SIZE, n_out = 1, kernel_size = KERNEL_SIZE, 
                    dilations = DILATIONS) 




class SpiralnetRAVEClassifierGRU(nn.Module):
    def __init__(self, nr_of_classes, embedding_dim=N_BAND, nr_spiralnet_layers=4, nr_rnn_layers=2):
        super(SpiralnetRAVEClassifierGRU, self).__init__()
        self.nr_of_gesture_classes = nr_of_classes
        self.embedding_dim = embedding_dim
        self.audio_encoder = encoder
        self.pqmf = pqmf 
        self.spiralnet = instantiate_spiralnet(nr_layers=nr_spiralnet_layers, output_dim= self.embedding_dim)
        self.layer_norm = nn.LayerNorm(2 * self.embedding_dim)
        self.gru = nn.GRU(2 * self.embedding_dim, 2 * self.embedding_dim, nr_rnn_layers, bidirectional=False, batch_first=False)
        self.gelu = nn.GELU()
        self.output_values_ff_list = nn.ModuleList([nn.Linear(2 * self.embedding_dim, 5) 
                                                    for _ in range(self.nr_of_gesture_classes)])
        self.fc = nn.Linear(2 * self.embedding_dim, self.nr_of_gesture_classes)
        self.softmax = nn.Softmax(dim=0)
        

        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)

    def forward(self, pose_tensor, audio_buffer):
        pose_embedding = self.spiralnet(pose_tensor)
        pqmf = self.pqmf(audio_buffer)
        audio_embedding = self.audio_encoder(pqmf).squeeze(2)
        x = torch.concat((pose_embedding, audio_embedding), dim=1)
        #print('x.shape', x.shape)
        x = self.layer_norm(x)
        x, _ = self.gru(x)
        x = self.gelu(x[-1])
        logits = self.fc(x)  # use this as joint embedding for the MINE
        class_prediction = torch.argmax(logits).item()
        values = self.output_values_ff_list[class_prediction](x)
        softmax_values = self.softmax(values)
        #values = 127 * torch.sigmoid(values) 
        downsamples_embedding_for_transmission = F.avg_pool1d(pqmf, kernel_size=4, stride=4)
        return logits, class_prediction, softmax_values, downsamples_embedding_for_transmission

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



class GestureApp:
    def __init__(self):
        
        
        # Initialize variables
        self.capturing = False
        self.gestures = {}
        self.model = SpiralnetRAVEClassifierGRU(10) #None§
        print('model number of parameters = ', sum(p.numel() for p in self.model.parameters() if p.requires_grad) )
        #self.state = "initial_state"
        self.state = "prediction_state"
        self.pose_buffer = deque(maxlen=45)
        self.assets = {'model': self.model} 
        self.criterion = nn.CrossEntropyLoss()
        self.current_nr_of_classes = 0
        self.nr_class_labels = 0
        

        #self.mp_holistic = mp.solutions.holistic(min_detection_confidence=0.5, 
        #                                                  min_tracking_confidence=0.5,
        #                                                  model_complexity=0)  # Using a lighter model
        self.cap = cv2.VideoCapture(0)
        self.cap.set(3, 640)  # Set resolution width
        self.cap.set(4, 480)  # Set resolution height



        # Initialize Video Capture
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(3))
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            exit()

        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(min_detection_confidence=0.5, 
                                               min_tracking_confidence=0.5)

        self.mp_drawing = mp.solutions.drawing_utils


        # Initialize variables
        self.capturing = False
        self.gestures = {}
        self.current_gesture = deque(maxlen=2)
        self.osc_input_port = osc_input_port
        self.osc_output_port = osc_output_port
        self.training_output_port = 8004
        self.osc_client = udp_client.SimpleUDPClient(osc_output_ip, osc_output_port)#('146.70.72.139', 42000)#('127.0.0.1', 11000)  # IP and port for Ableton Live
        self.osc_client_training_data = udp_client.SimpleUDPClient(osc_output_ip, self.training_output_port )
        self.check_output = {} 
        self.reward_labels = {}
        self.current_prediction = None
        self.current_logits_max = 0
        

        self.midiout = rtmidi.MidiOut()
        self.available_ports = self.midiout.get_ports()
        print(self.available_ports)
        self.midiout.open_port(0)

        self.pitch_classes = {
            0: [60, 62, 64, 67, 69],  # C Major Pentatonic
            1: [67, 69, 71, 74, 76],  # G Major Pentatonic
            2: [62, 64, 66, 69, 71],  # D Major Pentatonic
            3: [69, 71, 73, 76, 78],  # A Major Pentatonic
            4: [64, 66, 68, 71, 73],  # E Major Pentatonic
            5: [71, 73, 75, 78, 80],  # B Major Pentatonic
            6: [65, 67, 69, 72, 74],  # F Major Pentatonic
            7: [72, 74, 76, 79, 81],  # C# Major Pentatonic
            8: [66, 68, 70, 73, 75],  # F# Major Pentatonic
            9: [73, 75, 77, 80, 82], # G# Major Pentatonic
            10: [68, 70, 72, 75, 77], # Bb Major Pentatonic
            11: [75, 77, 79, 82, 84], # Eb Major Pentatonic
        }

        self.last_pose = None
        #self.audio_queue = queue.Queue()
        self.audio_buffer_size = 2048#4096
        self.buffer_size = 8#8  # Example size, adjust as needed
        self.audio_buffer = deque(maxlen=self.buffer_size)#np.zeros((self.buffer_size, self.audio_buffer_size), dtype=np.float32)
        self.write_index = 0
        self.read_index = 0
        self.current_pose_read = None
        self.current_pose_write = torch.zeros(33, 3)
        self.lock = threading.Lock() 
        self.pose_count = 0
        self.audio_buffer_count = 0
        self.poses_deque = deque(maxlen=self.buffer_size)
        self.audio_buffers_and_pose_counts_deque = deque(maxlen=1000)
        self.current_audio_frame_read = 0
        self.current_audio_frame_play = 0
        self.random_test_pose_data = torch.rand(10000, 33, 3)
        
        
        self.audio_buffers_and_pose_counts_deque.append([np.zeros((1, 1, self.audio_buffer_size), dtype=np.float32), self.audio_buffer_count, self.pose_count])
        self.model.eval()
        # Start the asynchronous processing thread
        self.device = torch.device('cpu')#torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        print('device = ', self.device)
        self.model.to(self.device)
        self.input_audio_tensor = torch.zeros(1, 1, self.audio_buffer_size).to(self.device)#
        self.input_pose_tensor = torch.zeros(33, 3).to(self.device)
        self.poses_deque.append(self.input_pose_tensor)
        #self.processing_thread = threading.Thread(target=self.process_audio)
        #self.processing_thread.start()


    def audio_callback(self, in_data, frame_count, time_info, status):
        print('audio callback ', self.audio_buffer_count)#, flush=True)
        audio_data = np.frombuffer(in_data, dtype=np.float32)
        print(audio_data)
        self.audio_buffers_and_pose_counts_deque.append([audio_data, self.audio_buffer_count, self.pose_count])  # Appending the new audio data to the right end of the deque
        self.audio_buffer_count +=1
        #print('audio callback ', self.current_audio_frame_read, flush=True)
        return (in_data, pyaudio.paContinue)
    
    def agent_model_inference(self, audio_buffer, pose):
        #model = self.assets['model']
        with torch.no_grad():
            #print('type(audio_buffer)', type(audio_buffer))
            #if isinstance(audio_buffer, np.ndarray):
            # Reshape and convert audio_buffer to a tensor, then update input_audio_tensor
            audio_buffer = np.copy(audio_buffer) # unfortunately need to do a copy since the audio buffer is not writeable
            
            self.input_audio_tensor[:] = torch.from_numpy(audio_buffer).view(1, 1, -1).float()#.to(self.device)
                
            #else:
            #    raise ValueError("Expected audio_buffer to be a numpy array")

            #self.input_audio_tensor[:] = torch.from_numpy(audio_buffer).view(1, 1, -1).float()
            audio_tensor, pose_tensor = self.input_audio_tensor, pose#.to(self.device)
            #print('audio tensor shape : ', audio_tensor.shape, 'pose tensor shape : ', pose_tensor.shape)
            #ti = time.time()
            logits, class_prediction, softmax_values, input_audio_embedding = self.model(pose_tensor, audio_tensor)
            #print(logits)
            #ti = time.time()-ti
            #print('inference time in func = ', ti)
            self.current_logits_max = torch.max(logits)
            self.map_midi_to_ableton(class_prediction, softmax_values) 
        return logits, class_prediction, softmax_values, input_audio_embedding


    def transmit_training_data(self, logits, class_prediction, softmax_values, pose_count, audio_buffer_count, input_audio_embedding):
        logits_list = logits.flatten().tolist()
        # Handle class_prediction based on its type
        if isinstance(class_prediction, int):
            class_prediction_list = [class_prediction]  # Directly use the integer value
        else:
            class_prediction_list = class_prediction.flatten().tolist()  # If it's tensor or array

        softmax_values_list = softmax_values.flatten().tolist()

        input_audio_list = input_audio_embedding.flatten().tolist()

        data_list = [
            pose_count, audio_buffer_count,
            len(logits_list), *logits_list,
            len(class_prediction_list), *class_prediction_list,
            len(softmax_values_list), *softmax_values_list, 
            len(input_audio_list), *input_audio_list
        ]

        self.osc_client_training_data.send_message("/training_data", data_list)

    def map_agent_outputs(self, *outputs):
        # depending on output prediction class, logits distribution, action truth value, timing value
        pass

    
    def map_midi_to_ableton(self, prediction, value):
        self.send_notes_async(prediction, value)
        pass


    def send_notes(self, pitch_class, distribution):
        number_of_notes = len(distribution)

        scale = self.pitch_classes[pitch_class]

        # Number of notes to play
        number_of_notes = len(distribution)


        for i in range(number_of_notes):
            pitch = random.choices(scale, distribution, k=1)[0]

            # Sending MIDI messages
            note_on = [0x90, pitch, 100]  # Note On message
            note_off = [0x80, pitch, 0]   # Note Off message

            self.midiout.send_message(note_on)
            time.sleep(0.1)  # Adjust as needed
            self.midiout.send_message(note_off)

    def send_notes_async(self, pitch_class, distribution):
        thread = threading.Thread(target=self.send_notes, args=(pitch_class, distribution))
        thread.start()

    def init_osc_server(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/feedback", self.update_reward_label)
        self.dispatcher.map("/pose", self.handle_poses)
        self.osc_ip = "127.0.0.1"  # Replace with your OSC server IP
        self.osc_port = self.osc_input_port       # Replace with your OSC server port
        self.osc_server = BlockingOSCUDPServer((self.osc_ip, self.osc_port), self.dispatcher)

    def start_osc_server(self):
        print("Starting OSC Server")
        self.osc_server.serve_forever()  # This will block until the server is stopped

    def stop_osc_server(self):
        if self.osc_server:
            self.osc_server.shutdown()
            print("OSC Server stopped")

    def update_reward_label(self, unused_addr, *args):
        self.reward_labels[self.current_prediction].append([args[0], self.current_logits_max])
        print("Received OSC message, reward label of current prediction =:", args[0])

    def handle_poses(self, unused_addr, *args):
        
        self.input_pose_tensor = torch.tensor(args).view(33, 3)
        self.poses_deque.append(self.input_pose_tensor)
        self.pose_count+=1
        #print(len(args), self.pose_count)
        pass


    def store_as_torch_tensors(self, gestures):
        for key in gestures.keys():
            for i in range(len(gestures[key])):
                numpy_array = np.array(gestures[key][i], dtype=np.float32)
                torch_tensor = torch.from_numpy(numpy_array)
                gestures[key][i] = torch_tensor
    
    def start_audio_stream(self):
        audio = pyaudio.PyAudio()

        # List all available audio devices
        info = audio.get_host_api_info_by_index(0)
        num_devices = info.get('deviceCount')
        for i in range(0, num_devices):
            if (audio.get_device_info_by_host_api_device_index(0, i).get('maxInputChannels')) > 0:
                device_info = audio.get_device_info_by_host_api_device_index(0, i)
                print("Input Device id ", i, " - ", device_info.get('name'))

        # Open an audio stream with a specific input device
        stream = audio.open(format=pyaudio.paFloat32,
                            channels=1,
                            rate=44100,
                            input=True,
                            frames_per_buffer=self.audio_buffer_size,
                            stream_callback=self.audio_callback,
                            input_device_index=1)  # Set input device index to 2

        stream.start_stream()
        print('audio stream started')
        return stream, audio



    def capture_gestures_and_inference(self):

        #osc server : 
        # dispatcher 1: receive poses osc messages from e.g mediapipe poses input app
        # dispatcher 2: receive feedback values from RLHF app
        # todo : (i) create realtime training infrastructure on linux gpu pc, transfering model statedict over local network to update 
        # the model in inference. 
        # (ii) implement model transfer functionality in inference app, to load the most recently received model state dict, 
        # and compiling and providing the training data for the training server in realtime over local network.

        print("Starting ableton agent")
        #self.audio_thread = threading.Thread(target=self.start_audio_stream)
        self.osc_thread = threading.Thread(target=self.start_osc_server)
        #self.audio_thread = threading.Thread(target=self.start_audio_stream)
        
        #elf.audio_thread.start()
        self.osc_thread.start()
        self.start_audio_stream()
        #self.audio_thread.start()
        #self.start_osc_server()
        
        #self.osc_client_training_data.send_message("/testing", 0)

        while True:
            print('while loop started')
            #t = time.time()
            
            audio_buffer, audio_buffer_count, pose_count = self.audio_buffers_and_pose_counts_deque[-1]
            pose = self.poses_deque[pose_count]
            print(pose)
            #print(audio_buffer)
            #print("Type of audio buffer:", type(audio_buffer), flush=True)
            #print("Content of deque:", [type(item[0]) for item in list(self.audio_buffers_and_pose_counts_deque)], flush=True)

            logits, class_prediction, softmax_values, input_audio = self.agent_model_inference(audio_buffer, pose)

            self.transmit_training_data(logits, class_prediction, softmax_values, pose_count, audio_buffer_count, input_audio)
            #self.current_audio_frame_play = self.current_audio_frame_read
            #t = time.time()-t
            #print('agent inference time = ', t)


    def send_mod(self, cc=1, value=0):
        mod1 = ([CONTROL_CHANGE | 0, cc, value])
        print(value)
        if value > 0.0:
            self.midiout.send_message(mod1)

    def play_a_clip(self, track_number, clip_number):
        client_path = "/live/clip/fire"
        self.osc_client.send_message(client_path, [track_number, clip_number])

        # Stop all clips
    def stop_all_clips(self):
        client_path = "/live/song/stop_all_clips"
        self.osc_client.send_message(client_path, None)
        print("Stop all clips!")

    def stop_playing(self):
        client_path = "/live/song/stop_playing"
        self.osc_client.send_message(client_path, None)
        print("Stop playing!")

    # Set the volume of a clip in Ableton Live
    def set_clip_volume(self, track_number, clip_number, volume):
        """
        Adjusts the volume of a specified clip in a specified track in Ableton Live.

        Parameters:
        - track_number (int): The index of the track (0-based).
        - clip_number (int): The index of the clip (0-based).
        - volume (float): The desired volume (gain) level. Typically between 0.0 and 1.0.
        """
        clip_path = f"/live/track/set/volume"
        self.osc_client.send_message(clip_path, [1, volume])

    # set the frequency of the autofilter
    def set_autofilter_frequency(self, value):
        client_path = "/live/device/set/parameter/value"
        self.osc_client.send_message(client_path, [0, 1, 5, value])

    def set_autofilter_device_active(self, value):
        v = 1.0 if value else 0.0
        client_path = "/live/device/set/parameter/value"
        self.osc_client.send_message(client_path, [0, 1, 0, v])

    def arm_a_track(self, trackId, value):
        v = 1.0 if value else 0.0
        client_path = "/live/track/set/arm"
        self.osc_client.send_message(client_path, [2, v])

    # Send OSC message to Ableton Live
    def send_osc_to_ableton(self, note, velocity=64):
        # client = udp_client.SimpleUDPClient("127.0.0.1", 9000)  # Default port for LiveOSC

        # In this example, we'll trigger a note in a MIDI track in Ableton
        # You might need to adapt paths or actions based on your LiveOSC and Ableton setup

        # Assuming track 1 is a MIDI track
        track_path = "/live/track/view/1"

        # Start a clip in the first slot of the track to play a MIDI note
        clip_path = track_path + "/clip/view/1"
        self.osc_client.send_message(clip_path + "/notes", [note, velocity, 100])  # 100ms length for the note
        self.osc_client.send_message(clip_path + "/deselect_all_notes", [])
        self.osc_client.send_message(clip_path + "/start_playing", [])
        

    def initial_state(self):
        self.capture_initial_gestures()
        pass

    def prediction_state(self):
        print('prediction state initialised successfully')
        # Implementation for prediction state
        self.capture = True
        self.capture_gestures_and_inference()
        pass

    def training_state(self):
        self.RLHF_training()
        pass


    #def training_state(self):
    #    self.train_model()
        # Implementation for training state
    #    pass


    def run(self):
        # Initialize OSC server
        self.init_osc_server()

        # Start OSC server in a separate thread
        osc_thread = threading.Thread(target=self.start_osc_server)
        osc_thread.start()

        # Main application loop
        while True:
            # ... your application logic here ...
            if self.state == "initial_state":
                self.initial_state()
            elif self.state == "prediction_state":
                self.prediction_state()
            elif self.state == "training_state":
                self.training_state()
            else:
                print("Invalid state. Exiting.")
                break

        # Cleanup
        self.cleanup()
        self.stop_osc_server()
        osc_thread.join()

if __name__ == "__main__":
    app = GestureApp()
    app.run()
