import numpy as np

def compute_distance(landmark1, landmark2, image_w, image_h):
    """Compute Euclidean distance between two landmarks."""
    x1, y1 = landmark1.x * image_w, landmark1.y * image_h
    x2, y2 = landmark2.x * image_w, landmark2.y * image_h
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)

# takes landmark value and returns decimal for volume control:
def scale_to_range(value, input_min, input_max, output_min, output_max):
    """
    Scale `value` from the range [input_min, input_max] to [output_min, output_max].

    Parameters:
    - value (float): The input value to scale.
    - input_min (float): Minimum of the input range.
    - input_max (float): Maximum of the input range.
    - output_min (float): Minimum of the desired output range.
    - output_max (float): Maximum of the desired output range.

    Returns:
    - float: The scaled value.
    """
    return output_min + ((value - input_min) / (input_max - input_min)) * (output_max - output_min)

def map_to_decimal(value):
    """Map value from range [1, 1050] to [0.1, 1]."""
    return scale_to_range(value, 1, 1050, 0.1, 1)

def convert_range(value, in_min, in_max, out_min, out_max):
    l_span = in_max - in_min
    r_span = out_max - out_min
    scaled_value = (value - in_min) / l_span
    scaled_value = out_min + (scaled_value * r_span)
    return np.round(scaled_value)