import torch
import cv2
import mediapipe as mp
import numpy as np
import threading
from collections import deque
from pythonosc import dispatcher, osc_server
from models import SpiralnetClassifierGRU
from model_inference_manager import ModelManager
#from modelmanager2 import ModelManager
from pythonosc import udp_client
from math import prod

# Initialize OSC client (adjust IP and port as needed)
ip_send = "192.168.1.103" #"192.168.1.1"
port_send = 8005  # Change to the port where you want to send data

# OSC server setup
ip_receive = "0.0.0.0"#"192.168.1.103"#"192.168.1.2"
port_receive = 8082

ip_ableton = "127.0.0.1"
port_ableton = 10013

osc_client = udp_client.SimpleUDPClient(ip_send, port_send)

ableton_client = udp_client.SimpleUDPClient(ip_ableton, port_ableton)

#osc_client.send_message('/test', [])
print('test message sent to the PC')

# Initialize Model Manager


model_manager = ModelManager(SpiralnetClassifierGRU, 10)
# Initialize variables for pose landmarks
pose_buffer = deque(maxlen=45)  # Adjust the length based on your model's input
max_chunk_size = 1024
received_params = {}
all_params = []


#At the end of process, check :
#condition 1: has shape 

#When all chunks for a param has been sent form the training server, 
#and /param_chunks_transmitted has beed received on the client, check;
#condition 2: 
# or 2.5: if param is sent in one chunk; is of total length shape[2] 
#condition 3: then order chunks according to their chunk_nr list,
#stack the chunks to a torch tensor and copy it to model.parameters()

def handle_param_shape(address, *args):
    param_index = args[0]
    param_shape = args[1:]
    #print('param_index : ', param_index)
    #print('param_shape : ', param_shape)
    if param_index not in received_params.keys():
        received_params[param_index] = [[], [], param_shape, False]

def handle_model_param_full(address, *args):
    # case 1 : the whole parameter comes in one chunk
    param_index = args[0]
    #print('full param received with ind ', param_index)
    shape_len = args[1]
    param_shape = args[2:2+shape_len] # the len of shape is stored in the second dimension
    param_list = args[2+shape_len:]
    received_params[param_index] = [param_list, param_index ,param_shape, False]


def handle_param_chunk(address, *args):
    param_index = args[0]
    #print('chunk received for param ', param_index)
    chunk_nr = args[1]
    shape_len = args[2]
    param_shape = args[3:3+shape_len]
    chunk = args[3+shape_len:]
    if param_index not in received_params.keys():
        received_params[param_index] = [[], [], param_shape, False]
    received_params[param_index][0].append(chunk)
    #print('param_index ', param_index, ' received chunk len ', len(chunk), ' chunk nr ', chunk_nr )
    received_params[param_index][1].append(chunk_nr)
     # then by the end order the chunks according to their chunk nr's list, 
     # and if last chunk is shorter, first make tensor of all except the last chunk, 
     # then append the tensored last chunk to the tensor, and append it together with 
     # it's shape and index to all params list

def handle_last_chunk(address, *args):
    # set last unequally space chunk recived to True, to use in update parameters call later
    #print('last chunk received ')
    param_index = args[0]
    received_params[param_index][3] = True 
    chunk_nr = args[1]
    shape_len = args[2]
    param_shape = args[3:3+shape_len]
    chunk = args[3+shape_len:]
    if  prod(param_shape) > 0 and max_chunk_size * chunk_nr >= prod(param_shape):
        received_params[param_index][0].append(chunk)
        received_params[param_index][1].append(chunk_nr)
        sorted_chunks = [received_params[param_index][0][i] for i in received_params[param_index][1]]
        #sorted_chunks = [x for _, x in sorted(zip(received_params[param_index][1], received_params[param_index][0]))]
        received_params[param_index][0] = sorted_chunks
    else:
        print('incomplete chunk received')

def all_chunks_transmitted(address, *args):
    
    param_index = args[0]
    #print('all chunks transmitted for param ', param_index)
    chunk_nr = args[1]
    param_shape = args[2:]
    
    if prod(param_shape) > 0 and max_chunk_size * chunk_nr >= prod(param_shape):
        sorted_chunks = [received_params[param_index][0][i] for i in received_params[param_index][1]]
        #sorted_chunks = [x for _, x in sorted(zip(received_params[param_index][1], received_params[param_index][0]))]
        received_params[param_index][0] = sorted_chunks
    else:
        print('incomplete chunk received ')


def handle_end_of_transmission(address, *args):
    print("Received end of transmission signal. Received parameters data : ")
#    print(' ########################################################')
#    for key, value in received_params.items():
#        
#        print('param index : ', key)
#        if hasattr(value[0], '__len__'):
#            print('param nr of chunks : ', len(value[0]))
#            print('param nr of chunk numbers : ', len(value[1]))
#
     #   else:
    #        
    #        print('not len attribute, value[0] = ', value[0])
    #    print('param shape : ', value[2])
    #    print('if unequal last chunk : ', value[3])
    #    print(' -------------------------------------------------- ')
    #print(' ########################################################')

    # Update the model parameters
    model_manager.update_parameters(received_params)

    

# The rest of the OSC server setup remains the same

# Setup your OSC server
dispatcher = dispatcher.Dispatcher()
dispatcher.map("/param_shape", handle_param_shape)
dispatcher.map("/param_chunk", handle_param_chunk)
dispatcher.map("/param_transmitted_full", handle_model_param_full)
dispatcher.map("/all_chunks_transmitted", all_chunks_transmitted)
dispatcher.map("/last_chunk", handle_last_chunk)
dispatcher.map("/end_params_transmission", handle_end_of_transmission)



def run_osc_server():
    server = osc_server.ThreadingOSCUDPServer((ip_receive, port_receive), dispatcher)
    server.serve_forever()

# Start OSC server in a separate thread
server_thread = threading.Thread(target=run_osc_server)
server_thread.start()

# Initialize webcam
# Setup MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils
cap = cv2.VideoCapture(0)

# Main loop for pose detection and model inference
with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)

        # Draw pose landmarks
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)

        # Process pose landmarks
        try:
            pose = results.pose_landmarks.landmark
            # Ensure the pose_row has the shape (33, 3)
            pose_list = [value for landmark in pose for value in (landmark.x, landmark.y, landmark.z)]

            osc_client.send_message("/pose", list(pose_list))
            #print('send pose osc')
            #pose_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose])
            
            # Check if pose_row has 33 landmarks
            #if pose_row.shape[0] == 33:
                # Reshape pose_row and create a tensor of shape (1, 33, 3)
            pose_tensor = torch.tensor(pose_list, dtype=torch.float32).view(1, 33, 3)

                # Model inference
            with torch.no_grad():
                    active_model = model_manager.get_active_model()
                    output = active_model(pose_tensor)

                    # Convert output to list of floats and send OSC message
                    output_list = output.squeeze().tolist()  # Adjust based on your model's output structure
                    #print(output_list)
                    ableton_client.send_message("/model_output", output_list)

        except AttributeError:
            pass

        cv2.imshow('Pose Detection', cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
        if cv2.waitKey(1) & 0xFF == 27:  # ESC to exit
            break

cap.release()
cv2.destroyAllWindows()
