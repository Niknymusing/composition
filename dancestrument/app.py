import cv2
import mediapipe as mp
import numpy as np

from utils.autotune import play_in_scale
from utils.ableton_functions import play_a_clip, set_autofilter_frequency, stop_all_clips, set_autofilter_device_active, stop_playing, arm_a_track
from utils.midi_functions import send_notes, send_mod
from utils.ais_little_helpers import compute_distance, scale_to_range, convert_range
from utils.landmark_definitions import landmarks_dict, right_hand_landmarks_dict, left_hand_landmarks_dict
import pickle
import pandas as pd


def main():

    # Prepare Camera
    cap = cv2.VideoCapture(0)

    # Load models
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose
    mp_holistic = mp.solutions.holistic

    # Load Custom Trained Model (Uncomment to use your custom-trained model)
    with open('./train/pose_model.pkl', 'rb') as f:
        model = pickle.load(f)

    last_note = None

    # Main loop
    with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
        while cap.isOpened():

            # Process key to exit: ##################
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

            # Camera Capture ##################
            ret, frame = cap.read()
            if not ret:
                print("Ignoring empty camera frame.")
                continue

            # Make detections from frame ##################
            image = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB) # Convert the BGR image to RGB and flip.
            image_h, image_w, _ = image.shape
            
            results = holistic.process(image)
            image.flags.writeable = True # Draw the pose annotation on the image.

            # Recolor back to BGR for rendering on screen.
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # 1. Pose Detections
            mp_drawing.draw_landmarks(
                image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
            
            # If we have pose detection results =================
            if results.pose_landmarks:
                boundary_landmarks = [
                    results.pose_landmarks.landmark[i] for i in [11, 12, 23, 24, 27, 28]
                ]

                for idx, landmark in enumerate(results.pose_landmarks.landmark):
                    # Check landmark visibility
                    if landmark.visibility < 0.5:  # You can adjust this threshold as needed.
                        continue

                    landmark_x = min(int(landmark.x * image_w), image_w - 1)
                    landmark_y = min(int(landmark.y * image_h), image_h - 1)
                    
                    # Adds labels to landmarks
                    # draw_landmark_labels(landmark_x, landmark_y, idx, image, landmarks_dict)

                    # Draw landmark points on the image
                    left_shoulder = results.pose_landmarks.landmark[12]
                    right_shoulder = results.pose_landmarks.landmark[11]

                    right_wrist = results.pose_landmarks.landmark[15]
                    right_wrist_x = results.pose_landmarks.landmark[15].x * image_w
                    
                    left_wrist = results.pose_landmarks.landmark[16]
                    left_wrist_x = results.pose_landmarks.landmark[16].x * image_w

                    left_pinky = results.pose_landmarks.landmark[18]
                    right_pinky = results.pose_landmarks.landmark[17]
                    right_hip = results.pose_landmarks.landmark[23]

                    distance_1 = compute_distance(right_wrist, left_shoulder, image_w, image_h)
                    distance_2 = compute_distance(left_wrist, right_shoulder, image_w, image_h)
                    distance_3 = compute_distance(left_wrist, right_hip, image_w, image_h)
                    distance_between_wrists = compute_distance(left_wrist, right_wrist, image_w, image_h)

                    if right_wrist.visibility >= 0.5 and left_wrist.visibility >= 0.5:
                        if round(distance_1) < 100:
                            # play_a_clip(5, 0)
                            print("Play drums")
                        if round(distance_2) < 100:
                            # play_a_clip(6, 1)
                            print("Play bass")
                        if round(distance_3) < 100:
                            # stop_all_clips()
                            print("Stop all clips")
                        if round(distance_between_wrists) < 10:
                            # play_a_clip(8, 0)
                            print("Play crash")
                    
                    # if right wrist appears on right side of screen
                    if right_pinky.visibility >= 0.8:
                        if right_pinky.x > 0.5:
                            if round(right_pinky.x * image_h) in range(1, 720):
                                print("Melody")
                                v2 = convert_range(right_pinky.y, 1.0, -1.0, 60, 92)
                                v2 = play_in_scale(v2, "nonatonic")
                                print(v2)
                                send_notes(v2, 1)
                                # if v2 != last_note:
                                #     send_notes(v2, 1)
                                #     # Update the list of the last two notes
                                #     last_note = v2


                    
                    if left_pinky.visibility >= 0.8:
                        if left_pinky.x < 0.5: # left side of screen
                            set_autofilter_device_active(True)
                            if round(left_pinky.y * image_h) in range(1, 720):
                                
                                v = left_pinky.y * image_h
                                print(v)
                                v2 = convert_range(v, 720.0, 1.0, 20, 135)
                                # freqValue = map_to_decimal(v)
                                # freqValue = v * 10
                                # print(freqValue)
                                # freqValue = freqValue + 100
                                # mtd = map_to_decimal(v)
                                # print(v)
                                # set_clip_volume(1, 0, mtd)
                                set_autofilter_frequency(v2)
                        else:
                            set_autofilter_device_active(False)
                    else:
                        set_autofilter_device_active(False)


                    distance_x = round(abs(right_wrist_x - left_wrist_x))
                    rcd_x = round(distance_x / 100)
                    

                # Calculate the chest/chestbone position
                right_shoulder = results.pose_landmarks.landmark[11]
                left_shoulder = results.pose_landmarks.landmark[12]
                
                chest_x = (right_shoulder.x + left_shoulder.x) / 2
                chest_y = (right_shoulder.y + left_shoulder.y) / 2
                chest_x_pixel = int(chest_x * image_w)
                chest_y_pixel = int(chest_y * image_h)
            
                # Draw the chest point on the image (let's use a blue color for differentiation)
                cv2.circle(image, (chest_x_pixel, chest_y_pixel), 5, (255, 0, 0), -1)  # Blue color, filled circle

                # 2. Right hand
                mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(80,22,10), thickness=1, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(80,44,121), thickness=1, circle_radius=1)
                                        )

                # 3. Left Hand
                mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                        mp_drawing.DrawingSpec(color=(121,22,76), thickness=1, circle_radius=2),
                                        mp_drawing.DrawingSpec(color=(121,44,250), thickness=1, circle_radius=1)
                                        )

                # ============== Triggering with pose detections =================
                # Extract Pose landmarks
                pose = results.pose_landmarks.landmark
                pose_row = list(np.array([[landmark.x, landmark.y, landmark.z, landmark.visibility] for landmark in pose]).flatten())
                
                # Concate rows
                row = pose_row 

                # Make Detections
                X = pd.DataFrame([row])
                body_language_class = model.predict(X)[0]
                body_language_prob = model.predict_proba(X)[0]
                print(body_language_class, body_language_prob)
                
                # Grab eye coords
                coords = tuple(np.multiply(
                                np.array(
                                    (results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].x, 
                                    results.pose_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_EYE].y))
                            , [640,480]).astype(int))

                # Trigger action if model detects movement above threshold:
                if np.any(body_language_prob > 0.9):
                    draw_identified_pose_class(image, body_language_class, coords)
                    if body_language_class == "ChangeInstrument":
                        # play_a_clip(5, 0)
                        arm_a_track(2, True)
                    if body_language_class == "LeftHandOut":
                        # stop_playing()
                        play_a_clip(5, 0)
                # // ============== End triggering with pose detections =================
                        
                    
                # press q to exit 
                frame_width = int(cap.get(3))
                cv2.putText(image, "press Q to quit", (frame_width // 2 - 20, 50),  # Adjust this for better positioning
                cv2.FONT_HERSHEY_SIMPLEX, 
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA)

                # Show screen:
                cv2.imshow('DANCESTRUMENT - Movement, Music & Machines', image)
            
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    stop_all_clips()
                    break
                

    cap.release()
    cv2.destroyAllWindows()

def draw_identified_pose_class(image, body_language_class, coords):
    cv2.rectangle(image, 
                                (coords[0], coords[1]+5), 
                                (coords[0]+len(body_language_class)*20, coords[1]-30), 
                                (245, 117, 16), -1)
    cv2.putText(image, body_language_class, coords, 
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

def draw_landmark_labels(landmark_x, landmark_y, idx, image, landmarks_dict):
    cv2.putText(image, landmarks_dict[idx], (landmark_x, landmark_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
    print(f'Landmark: {landmarks_dict[idx]}, Coordinates: ({landmark_x}, {landmark_y})')


if __name__ == "__main__":
    main()