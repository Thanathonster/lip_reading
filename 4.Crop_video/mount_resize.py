import cv2
import numpy as np

def resize_data(frame, x, y, w, h):
    """resize data lip image
    frame : frame image per video
    x : point x
    y : point y
    w : width image
    h : height image"""
    frame = frame[y:y+h,x:x+w,:]
    height, width, _ = frame.shape
    expand_height = 46 / height
    expand_width = 140 / width
    if expand_height > 3 and expand_width > 3:
        return None
    if expand_height <= expand_width:
        new_width = width * expand_height
        frame = cv2.resize(frame, (int(new_width),46), interpolation=cv2.INTER_CUBIC)
    elif expand_height > expand_width:
        new_height = height * expand_width
        frame = cv2.resize(frame, (140, int(new_height), ), interpolation=cv2.INTER_CUBIC)

    height, width, _ = frame.shape

    padding_height = 46 - height
    padding_width = 140 - width
    # Calculate the padding on the top, bottom, left, and right
    pad_top = padding_height // 2
    pad_bottom = padding_height - pad_top
    pad_left = padding_width // 2
    pad_right = padding_width - pad_left
    # Pad the image with black color on all sides
    frame = np.pad(frame, ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), mode='constant', constant_values=0)

    return frame