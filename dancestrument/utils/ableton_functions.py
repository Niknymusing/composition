from pythonosc.udp_client import SimpleUDPClient

# OSC Setup
ip = "127.0.0.1"
to_ableton = 11000
from_ableton = 11001
client = SimpleUDPClient(ip, to_ableton)

# Fire a clip in Ableton Live
def play_a_clip(track_number, clip_number):
    client_path = "/live/clip/fire"
    client.send_message(client_path, [track_number, clip_number])
    print("Fired a clip!")

# Stop all clips
def stop_all_clips():
    client_path = "/live/song/stop_all_clips"
    client.send_message(client_path, None)
    print("Stop all clips!")

def stop_playing():
    client_path = "/live/song/stop_playing"
    client.send_message(client_path, None)
    print("Stop playing!")

# Set the volume of a clip in Ableton Live
def set_clip_volume(track_number, clip_number, volume):
    """
    Adjusts the volume of a specified clip in a specified track in Ableton Live.

    Parameters:
    - track_number (int): The index of the track (0-based).
    - clip_number (int): The index of the clip (0-based).
    - volume (float): The desired volume (gain) level. Typically between 0.0 and 1.0.
    """
    clip_path = f"/live/track/set/volume"
    client.send_message(clip_path, [1, volume])

# set the frequency of the autofilter
def set_autofilter_frequency(value):
    client_path = "/live/device/set/parameter/value"
    client.send_message(client_path, [0, 1, 5, value])

def set_autofilter_device_active(value):
    v = 1.0 if value else 0.0
    client_path = "/live/device/set/parameter/value"
    client.send_message(client_path, [0, 1, 0, v])

def arm_a_track(trackId, value):
    v = 1.0 if value else 0.0
    client_path = "/live/track/set/arm"
    client.send_message(client_path, [2, v])

# Send OSC message to Ableton Live
def send_osc_to_ableton(note, velocity=64):
    # client = udp_client.SimpleUDPClient("127.0.0.1", 9000)  # Default port for LiveOSC

    # In this example, we'll trigger a note in a MIDI track in Ableton
    # You might need to adapt paths or actions based on your LiveOSC and Ableton setup

    # Assuming track 1 is a MIDI track
    track_path = "/live/track/view/1"

    # Start a clip in the first slot of the track to play a MIDI note
    clip_path = track_path + "/clip/view/1"
    client.send_message(clip_path + "/notes", [note, velocity, 100])  # 100ms length for the note
    client.send_message(clip_path + "/deselect_all_notes", [])
    client.send_message(clip_path + "/start_playing", [])