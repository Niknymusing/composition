import subprocess
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from models import SpiralnetClassifierGRU
from pythonosc import udp_client
import threading
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler
from collections import deque

device = torch.device('cuda')
data_lock = threading.Lock()
# OSC server script and client details
ip_send = "192.168.1.101"
port_send = 8082
client = udp_client.SimpleUDPClient(ip_send, port_send)

# Start OSC Server Script
subprocess.Popen(["python", "dataserver.py"])

model = SpiralnetClassifierGRU(10).to(device)
optimizer = optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
data_folder = "pose_data"


def send_model_parameters(model, client):
    print('sending parameter update')
    max_chunk_size = 1000
    param_index = -1  # Add an index for each parameter
    
    for param in model.parameters():
        param_index += 1
        param_shape = list(param.shape)
        client.send_message("/param_shape", [param_index] + param_shape)
        param_list = param.data.flatten().tolist()
        print('param list len : ', len(param_list))
       
        last_chunk = False
        # Include the index with the shape
        

        if len(param_list) <= max_chunk_size:
            # Include the index with the parameter data
            client.send_message("/param_transmitted_full", [param_index] + [len(param_shape)] + param_shape + param_list)
            #param_index += 1 
        else:
            chunk_nr = 0
            total_length = 0
            for i in range(0, len(param_list), max_chunk_size):
                chunk = param_list[i:i + max_chunk_size]
                # Include the index with each chunk
                if len(chunk) == max_chunk_size:
                    client.send_message("/param_chunk", [param_index] + [chunk_nr] + [len(param_shape)] + param_shape + chunk)
                    chunk_nr += 1 #then in the other end check if chunk_nr * chunk_size < param[index]_shape[2] to determine wether to discard the param index or include in the model parameter update
                    total_length += len(chunk)
                
                else:
                    last_chunk = True
                    client.send_message("/last_chunk", [param_index] + [chunk_nr] + [len(param_shape)] + param_shape + chunk) if hasattr(chunk, '__len__') else client.send_message("/last_chunk", [param_index] + [chunk_nr] + [len(param_shape)] + param_shape + [chunk])
                    print('last chunk sent')
                    #param_index += 1 
            if last_chunk == False:
                client.send_message("/all_chunks_transmitted", [param_index] + [chunk_nr] + param_shape)
                #param_index += 1 

         # Increment the index for the next parameter

    client.send_message("/end_params_transmission", [])



data_folder = "pose_data"
data_queue = deque(maxlen=500)

class NewFileHandler(FileSystemEventHandler):
    def on_created(self, event):
        if not event.is_directory:
            file_path = event.src_path
            pose_data = torch.load(file_path)
            os.remove(file_path)  # Remove the file after loading
            data_queue.append(pose_data)

def start_file_watcher():
    event_handler = NewFileHandler()
    observer = Observer()
    observer.schedule(event_handler, path=data_folder, recursive=False)
    observer.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        observer.stop()
    observer.join()

# Start file watcher in a separate thread
file_watcher_thread = threading.Thread(target=start_file_watcher, daemon=True)
file_watcher_thread.start()

def train(model, optimizer, criterion, device = 'cuda'):
    device = torch.device(device)
    while True:
        if data_queue:
            pose_data = data_queue.popleft()
            for data_point in pose_data:
                        #print('data_point.shape : ', data_point.shape)
                data_point = data_point.unsqueeze(0).to(device)  # Add batch dimension
                class_index = torch.randint(0, 10, (1,)).item()  # Get a random class index
                        #print('class_index.shape : ', class_index.shape)
                target = torch.tensor(class_index, dtype=torch.long).to(device)  # Correct target format
                        #print('target.shape : ', target.shape)
                optimizer.zero_grad()
                output = model(data_point)
                        #print('output.shape : ', output.shape)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                        # Send model parameters as OSC messages
                threading.Thread(target=send_model_parameters, args=(model, client), daemon=True).start()

                print(f" loss: {loss.item()}")

        #time.sleep(1)  # Check for new files every second

if __name__ == "__main__":
    train(model, optimizer, criterion)
