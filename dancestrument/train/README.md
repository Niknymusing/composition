# Training A Custom Pose Detection Model

This directory contains the code for training a custom pose detection model.

## Workflow

### 1. Capture and Label Poses

1. Open `python 1_capture_poses.py` and set the name of the pose you want to capture in the `class_name` variable. For example, "Namaste"
2. Then run the script. This will open a window that shows your webcam feed.
3. Press `q` to quit once you've captured your pose. The pose data will be saved to `data.csv`.
4. Repeat this process for each pose you want to capture.

### 2. Train Model

Next, train your model by running `python 2_train_model.py`. This will train a model using the data in `data.csv` and save the model to `pose_model.pkl`.

### 3. Test Pose Detection

To test your new model, run `python 3_test_detections.py`. This will open a window that shows your webcam feed and the pose predictions from your model.