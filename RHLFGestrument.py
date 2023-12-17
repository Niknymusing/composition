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
from pythonosc.osc_server import AsyncIOOSCUDPServer
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
#import asyncio
import threading

osc_output_ip = '127.0.0.1'
osc_output_port = 11000
osc_input_ip = '0.0.0.0'
osc_input_port = 8001

class SpiralnetClassifierGRU(nn.Module):
    def __init__(self, nr_of_classes, embedding_dim=32, nr_spiralnet_layers=4, nr_rnn_layers=2):
        super(SpiralnetClassifierGRU, self).__init__()
        self.nr_of_gesture_classes = nr_of_classes
        self.embedding_dim = embedding_dim
        self.spiralnet = instantiate_spiralnet(nr_layers=nr_spiralnet_layers, output_dim=self.embedding_dim)
        self.layer_norm = nn.LayerNorm(self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim, self.embedding_dim, nr_rnn_layers, bidirectional=False, batch_first=False)
        self.gelu = nn.GELU()
        self.fc = nn.Linear(self.embedding_dim, self.nr_of_gesture_classes)
        self.softmax = nn.Softmax(dim=1)

        for param in self.gru.parameters():
            if len(param.shape) >= 2:
                init.xavier_uniform_(param)

    def forward(self, x):
        x = self.spiralnet(x)
        x = self.layer_norm(x)
        x, _ = self.gru(x)
        x = self.gelu(x[-1])
        logits = self.fc(x)
        prediction = torch.argmax(logits).item()
        return logits, prediction


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
        # Initialize MediaPipe Holistic
        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils
        
        # Initialize variables
        self.capturing = False
        self.gestures = {}
        self.model = SpiralnetClassifierGRU #None
        self.state = "initial_state"
        self.pose_buffer = deque(maxlen=45)
        self.assets = {}
        self.criterion = nn.CrossEntropyLoss()
        self.current_nr_of_classes = 0
        self.nr_class_labels = 0
        
        # Initialize Video Capture
        self.cap = cv2.VideoCapture(0)
        self.frame_width = int(self.cap.get(3))
        if not self.cap.isOpened():
            print("Error: Could not open video capture.")
            exit()

        self.mp_holistic = mp.solutions.holistic
        self.mp_drawing = mp.solutions.drawing_utils

        # Initialize variables
        self.capturing = False
        self.gestures = {}
        self.current_gesture = deque(maxlen=45)
        self.osc_client = udp_client.SimpleUDPClient(osc_output_ip, osc_output_port)#('146.70.72.139', 42000)#('127.0.0.1', 11000)  # IP and port for Ableton Live
        self.check_output = {} 
        self.reward_labels = {}
        self.current_prediction = None
        self.current_logits_max = 0


    def init_osc_server(self):
        self.dispatcher = Dispatcher()
        self.dispatcher.map("/feedback", self.update_reward_label)
        self.osc_ip = "127.0.0.1"  # Replace with your OSC server IP
        self.osc_port = 8000        # Replace with your OSC server port
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
        print("Received OSC message, reward label of current prediction =:", self.reward_labels[self.current_prediction][:-1][0])


    def store_as_torch_tensors(self, gestures):
        for key in gestures.keys():
            for i in range(len(gestures[key])):
                numpy_array = np.array(gestures[key][i], dtype=np.float32)
                torch_tensor = torch.from_numpy(numpy_array)
                gestures[key][i] = torch_tensor


    def compile_gestures_dict_and_initialise_model(self):
        self.store_as_torch_tensors(self.gestures)
        dataset = GestureDataset(self.gestures)
        nr_of_classes = dataset.get_number_of_classes()
        print('nr_of_classes = ', nr_of_classes)
        
        if self.current_nr_of_classes != nr_of_classes: # if 'model' not in self.assets:
            model = SpiralnetClassifierGRU(nr_of_classes)
            optimizer = Adam(model.parameters(), lr=0.001)
            self.assets['model'] = model
            self.assets['optimizer'] = optimizer
            self.current_nr_of_classes = nr_of_classes
            
        else:
            model = self.assets['model']
            optimizer = self.assets['optimizer']
        
        return model, optimizer, dataset

  

    def train_model(self):
        model = self.assets['model']
        dataset = self.assets['dataset']
        criterion = self.criterion
        optimizer = self.assets['optimizer']
                
        num_epochs = 100
                
        for epoch in range(num_epochs):
            running_loss = 0.0
            for i in range(len(dataset)):
                gesture, label = dataset[i]
                gesture = gesture#.unsqueeze(0)  # Add batch dimension of 1
                label = torch.tensor([label], dtype=torch.long)
                
                optimizer.zero_grad()
                outputs, _ = model(gesture)
                loss = criterion(outputs.unsqueeze(0), torch.tensor([label], dtype=torch.long)) 
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                
                # Print running loss
                if i % 1 == 0:  # print every 10 samples
                    print(f'[Epoch {epoch + 1}, Sample {i + 1}] loss: {running_loss / 10:.3f}')
                    running_loss = 0.0

        print('Finished Training')
    
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
                logits, prediction_label = model(gesture) # outputs, predicted_reward = model(gesture)
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




    def play_a_clip(self, track_number, clip_number):
        client_path = "/live/clip/fire"
        self.osc_client.send_message(client_path, [track_number, clip_number])
        #print("Fired a clip!")
    
    def map_to_ableton(self, prediction):
        self.play_a_clip(4, prediction)# keys are 0, 1, 2, 3, .... map to correct ranges and addresses
        #addresses = ['1', '1', 'Device On', 'Filter Type', 'Filter Circuit - LP/HP', 'Filter Circuit - BP/NO/Morph', 'Slope', 'Frequency', 'Resonance', 'Morph', 'Drive', 'Env. Modulation', 'Env. Attack', 'Env. Release', 'LFO Amount', 'LFO Waveform', 'LFO Frequency', 'LFO Sync', 'LFO Sync Rate', 'LFO Stereo Mode', 'LFO Spin', 'LFO Phase', 'LFO Offset', 'LFO Quantize On', 'LFO Quantize Rate', 'S/C On', 'S/C Gain', 'S/C Mix']#['/address1', '/address2', '/address3', '/address4']
        #values = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
        #address = '' #addresses[prediction]
        #value = values[prediction]
        pass#return address, value


    # The key function of the app below:

    def evaluate_gesture(self, gesture):
        model = self.assets['model']
        gesture = np.array(gesture, dtype=np.float32)
        gesture_tensor = torch.tensor(gesture, dtype=torch.float32)
        with torch.no_grad():
            logits, prediction = model(gesture_tensor)
            self.current_prediction = prediction
            self.current_logits_max = torch.max(logits)
            #print('prediction : ', prediction)
            self.reward_labels[self.current_prediction].append([0, self.current_logits_max])
            #self.check_output[0] = output
            #print('output = ', output)
            #print('output.shape = ', output.shape)
             # map prediction to correct value for the ableton project here 
            self.map_to_ableton(prediction) 
            #print('prediction = ', prediction)
            #self.osc_client.send_message(address, value)  
            #self.osc_client.send_message('/ableton_live', prediction)  
        return prediction

    def capture_initial_gestures(self):
        self.capturing = False
        current_gesture = []
        gesture_key = None
        gesture_lists = {}
        message = ""
        train_model_flag = False

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

                if self.capturing:
                    try:
                        pose = results.pose_landmarks.landmark
                        pose_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose])
                        current_gesture.append(pose_row)
                    except AttributeError:
                        pass
                    cv2.putText(image, "Capturing... ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Press a key to select a gesture, 'Enter' to record, 'Space' to train", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                if gesture_key is not None and not self.capturing:
                    cv2.putText(image, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Movement, Music & Machines', image)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key to exit
                    break
                elif key != 255:  # A key is pressed
                    if key != ord('\r') and key != ord(' '):  # Not Enter or Spacebar
                        gesture_key = chr(key)
                        self.capturing = False
                        message = f"Initialised gesture {gesture_key}, press Enter to record example"
                        print(f"Key {gesture_key} pressed")
                    elif key == ord('\r'):  # Enter key is pressed
                        if self.capturing:
                            self.capturing = False
                            if gesture_key not in gesture_lists:
                                gesture_lists[gesture_key] = []
                                self.reward_labels[self.nr_class_labels] = deque(maxlen=45)
                                self.nr_class_labels += 1
                                
                            gesture_lists[gesture_key].append(current_gesture)
                            self.gestures[gesture_key] = gesture_lists[gesture_key]
                            current_gesture = []
                            print(f"Capture stopped. Gesture for key {gesture_key} saved.")
                        elif gesture_key is not None:
                            self.capturing = True
                            print("Capture started.")
                    elif key == ord(' '):  # Spacebar is pressed
                        if not self.capturing:
                            train_model_flag = True
                            print("Spacebar pressed, exiting loop to train model")
                            break  # Exit the while loop

        if train_model_flag:
            print("Training the model... ")
            model, optimizer, dataset = self.compile_gestures_dict_and_initialise_model()
            self.assets['model'] = model
            self.assets['optimizer'] = optimizer
            self.assets['dataset'] = dataset
            self.train_model()
            # Add your model training code here
            print('Finished Training')
            self.state = 'prediction_state'




    def RLHF_training(self):

        print('RHLF_training state started')
        self.capturing = False
        current_gesture = []
        gesture_key = None
        gesture_lists = {}
        message = ""
        train_model_flag = False

        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

                if self.capturing:
                    try:
                        pose = results.pose_landmarks.landmark
                        pose_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose])
                        current_gesture.append(pose_row)
                    except AttributeError:
                        pass
                    cv2.putText(image, "Capturing... ", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Press a key to select a gesture, 'Enter' to record, 'Space' to train", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                if gesture_key is not None and not self.capturing:
                    cv2.putText(image, message, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2, cv2.LINE_AA)

                cv2.imshow('Movement, Music & Machines', image)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key to exit
                    break
                elif key != 255:  # A key is pressed
                    if key != ord('\r') and key != ord(' '):  # Not Enter or Spacebar
                        gesture_key = chr(key)
                        self.capturing = False
                        message = f"Initialised gesture {gesture_key}, press Enter to record example"
                        print(f"Key {gesture_key} pressed")
                    elif key == ord('\r'):  # Enter key is pressed
                        if self.capturing:
                            self.capturing = False
                            if gesture_key not in gesture_lists:
                                gesture_lists[gesture_key] = []
                                self.reward_labels[self.nr_class_labels] = deque(maxlen=45)
                                self.nr_class_labels += 1
                                
                            gesture_lists[gesture_key].append(current_gesture)
                            self.gestures[gesture_key] = gesture_lists[gesture_key]
                            current_gesture = []
                            print(f"Capture stopped. Gesture for key {gesture_key} saved.")
                        elif gesture_key is not None:
                            self.capturing = True
                            print("Capture started.")
                    elif key == ord(' '):  # Spacebar is pressed
                        if not self.capturing:
                            train_model_flag = True
                            print("Spacebar pressed, exiting loop to train model")
                            break  # Exit the while loop

        if train_model_flag:
            print("Training the model... ")
            model, optimizer, dataset = self.compile_gestures_dict_and_initialise_model()
            self.assets['model'] = model
            self.assets['optimizer'] = optimizer
            self.assets['dataset'] = dataset
            self.train_model_RLHF()
            # Add your model training code here
            print('Finished Training')
            self.state = 'prediction_state'

    def capture_and_predict_gestures(self):
        #global capturing, current_gesture, cap
        print("Starting capture and predict gestures")
        self.osc_thread = threading.Thread(target=self.start_osc_server)
        self.osc_thread.start()
        frame_width = self.frame_width
        with self.mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
            
            should_exit = False
            while self.cap.isOpened():
                ret, frame = self.cap.read()
                if not ret:
                    print("Failed to grab frame")
                    break

                image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = holistic.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                self.mp_drawing.draw_landmarks(image, results.right_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.left_hand_landmarks, self.mp_holistic.HAND_CONNECTIONS)
                self.mp_drawing.draw_landmarks(image, results.pose_landmarks, self.mp_holistic.POSE_CONNECTIONS)

                try:
                    pose = results.pose_landmarks.landmark
                    pose_row = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose])
                    self.current_gesture.append(pose_row)
                    if len(self.current_gesture) == 45:

                        prediction = self.evaluate_gesture(self.current_gesture) 
                        

                        cv2.putText(image, f"Prediction: Gesture_{prediction}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                except AttributeError:
                    pass

                if self.capture:
                    cv2.putText(image, "Predicting gestures ", (frame_width // 2 - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA)
                else:
                    cv2.putText(image, "Press 'Enter' to start/stop capturing", (frame_width // 2 - 20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

                cv2.imshow('Movement, Music & Machines', image)
                key = cv2.waitKey(1) & 0xFF

                if key == ord(' '):  # Add this condition to switch back to initial state
                    self.stop_osc_server()
                    self.state = "training_state"
                    break
            if should_exit:
                return
        #cap.release()
        #cv2.destroyAllWindows()
    

    def initial_state(self):
        self.capture_initial_gestures()
        pass

    def prediction_state(self):
        print('prediction state initialised successfully')
        # Implementation for prediction state
        self.capture = True
        self.capture_and_predict_gestures()
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
