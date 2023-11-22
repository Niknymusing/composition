# Scales patterns for C
scales_patterns = {
    'pentatonic_minor': [0, 3, 5, 7, 10],
    'pentatonic_major': [0, 2, 4, 7, 9],
    'octatonic_hw': [0, 1, 3, 4, 6, 7, 9, 10],
    'octatonic_wh': [0, 2, 3, 5, 6, 8, 9, 11],
    'nonatonic': [0, 1, 2, 4, 5, 7, 9, 11, 12],
    'hexatonic': [0, 3, 5, 6, 7, 10],
    'ionian': [0, 2, 4, 5, 7, 9, 11],
    'dorian': [0, 2, 3, 5, 7, 9, 10],
    'phrygian': [0, 1, 3, 5, 7, 8, 10],
    'lydian': [0, 2, 4, 6, 7, 9, 11],
    'mixolydian': [0, 2, 4, 5, 7, 9, 10],
    'aeolian': [0, 2, 3, 5, 7, 8, 10],
    'locrian': [0, 1, 3, 5, 6, 8, 10],
}

def play_in_scale(midi_num, pattern):
    # Generate the full range scale
    scale = [note + 12 * octave for octave in range(11) for note in scales_patterns[pattern]]

    # Find the closest MIDI number in the scale
    return min(scale, key=lambda x: abs(x - midi_num))
