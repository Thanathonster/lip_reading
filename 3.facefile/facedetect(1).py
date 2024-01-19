# Are not completely 1,2,4,7,10
"""face detection are detection face and facealiment
if face detection are not detect not save that video"""
import numpy as np
import pandas as pd
import os
import json

import tqdm

import mediapipe as mp
from facedetection import face_detect


# add file
num = 1

floder = f"../data/Collect_{num}"
df = pd.read_csv(f"{floder}/Clean_length.csv")

# face-detection
mp_face_detection = mp.solutions.face_detection
face_det = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.3)
# face-landmark
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.3, min_tracking_confidence=0.3,refine_landmarks=True)

for path, video in tqdm.tqdm(zip(df["path"],df["video"])):
    # arrow the element datafile
    with open(f"{floder}/Record.json", "w", encoding='utf8') as outfile:
        json_object = {"path": path, "video": video}
        json.dump(json_object, outfile, ensure_ascii=False)

    try:
        face = f"{floder}/face/{path}"
        os.mkdir(face)
        os.mkdir(f"{face}/video")
        os.mkdir(f"{face}/aliface")
    except:
        pass

    try:
        num_people,yall, patch, roll = face_detect(f"{floder}/script/{path}/video/{video}", f"{face}/video",face_det,face_mesh)
        face_ali ={"people" : num_people,
                    "yall" : yall,
                    "patch" : patch,
                    "roll": roll}
        if set(num_people) == {0}: #check people in video
            continue
        with open(f'{face}/aliface/{video[:-4]}.json', 'w') as f:
            json.dump(face_ali, f)
    except:
        pass