# Dancestrument

## Description
Dancestrument is a musical instrument that can be played by a moving human body. It is a project for the hackathon, "Movement, Music & Machines". Questions? Contact [me](https://github.com/markschellhas).

## Installation

### Requirements
- Python 3.6
- pip
- virtualenv

### Setup
1. Clone this repository

2. Create a virtual environment
```
virtualenv -p python3 venv
```

3. Activate the virtual environment
```
source venv/bin/activate
```

4. Install the requirements
```
pip install -r requirements.txt
```
or

```
pip install opencv-python mediapipe numpy pandas python-osc python-rtmidi scikit-learn
```

### Run
Run the program
```
python app.py
```

## Usage

The `lib` directory contains the helper functions for communicating with Ableton Live via OSC and midi.

Copy the Ableton Live project in `/Example-1 \Project` to your computer and open it in Ableton Live.

### Train your own pose detection model

The `/train` directory contains the code for training a custom pose detection model.
Follow the jupyter notebook in `/train` to capture and label your own poses, train a model, and test the model.

To run the jupyter notebook, you will need jupyter installed.
```
pip install jupyter
```

Then run the notebook:
```
cd train
jupyter notebook
```