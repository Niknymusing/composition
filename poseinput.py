import numpy as np
from scipy.spatial.distance import cdist
import mediapipe as mp
import torch
from pythonosc import udp_client
import cv2
import argparse

# Argument parser for command line arguments
parser = argparse.ArgumentParser(description="Run pose tracking with OSC output.")
parser.add_argument("--port", type=int, default=4590, help="The port number for the OSC server.")
args = parser.parse_args()

# OSC client setup with dynamic port
client = udp_client.SimpleUDPClient('127.0.0.1', args.port)
mp_pose = mp.solutions.pose
mp_holistic = mp.solutions.holistic

cap = cv2.VideoCapture(0)

diff = 0.01
differentials = {0:0, 1:0, 2:0, 3:0}

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = holistic.process(image)


        if results.pose_landmarks:
            pose = results.pose_landmarks.landmark
            pose_array = np.array([[landmark.x, landmark.y, landmark.z] for landmark in pose])
            
            # Get the coordinates for left and right wrists
            right_wrist = pose_array[16, :].reshape(1, -1)
            left_wrist = pose_array[15, :].reshape(1, -1)
            nose = pose_array[0, :].reshape(1, -1)
            right_hip = pose_array[23, :].reshape(1, -1)
            left_hip = pose_array[24, :].reshape(1, -1)

            # Calculate the distance
            distance0 = cdist(right_wrist, left_hip, 'euclidean').item()
            distance1 = cdist(left_wrist, right_hip, 'euclidean').item()
            distance2 = cdist(nose, left_wrist, 'euclidean').item()
            distance3 = cdist(right_wrist, nose, 'euclidean').item()

            client.send_message("/latent_perturbations0", diff * np.sign(distance0 - differentials[0]))
            differentials[0] = distance0
            #print('message 0 sent', distance0)
            client.send_message("/latent_perturbations1", diff * np.sign(distance1 - differentials[1]))
            differentials[1] = distance1
            #print('message 1 sent', distance1)
            client.send_message("/latent_perturbations2", diff * np.sign(distance2 - differentials[2]))
            differentials[2] = distance2
            #print('message 2 sent', distance2)
            client.send_message("/latent_perturbations3", diff * np.sign(distance3 - differentials[3]))
            differentials[3] = distance3
            #print('message 3 sent', distance3)
            # Send the distance as an OSC message
            #client.send_message("/latent_perturbations0", distance0)
            #print('message 0 sent', distance0)
            #client.send_message("/latent_perturbations1", distance1)
            #print('message 1 sent', distance1)
            #client.send_message("/latent_perturbations2", distance2)
            #print('message 2 sent', distance2)
            #client.send_message("/latent_perturbations3", distance3)
            #print('message 3 sent', distance3)

            # For debugging
            #print(f"Distance between hands: {distance}")

        # Display the frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the capture
cap.release()
cv2.destroyAllWindows()
