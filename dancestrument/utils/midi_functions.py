import rtmidi
from rtmidi.midiconstants import (CONTROL_CHANGE)
import time
import random

# Midi Setup
midiout = rtmidi.MidiOut()
available_ports = midiout.get_ports()
print(available_ports)
midiout.open_port(0)


def send_notes(pitch=60, repeat=1):
    for i in range(repeat):
        note_on = [0x90, pitch, 112]
        note_off = [0x80, pitch, 0]
        midiout.send_message(note_on)
        # time.sleep(random.uniform(0.01, 0.8))
        midiout.send_message(note_off)

def send_mod(cc=1, value=0):
    mod1 = ([CONTROL_CHANGE | 0, cc, value])
    print(value)
    if value > 0.0:
        midiout.send_message(mod1)