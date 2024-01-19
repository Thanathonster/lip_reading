import cv2

def duration(filename):
    """find duration of video
    filename : path of videofile"""
    video = cv2.VideoCapture(filename)
    frame_count = video.get(cv2.CAP_PROP_FRAME_COUNT)

    return frame_count