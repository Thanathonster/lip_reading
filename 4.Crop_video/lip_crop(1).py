"""make video only lip"""
import cv2
import matplotlib.pyplot as plt
import numpy as np
import dlib as dlib
import os
import glob
import tqdm as tqdm
import json

from sklearn.cluster import KMeans
from sklearn.metrics import pairwise_distances

import mount_resize import resize_data

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


num = 1
floder = "../data"
face_flie = f"{floder}/Collect_{num}/face_1.json"
out_floder = f"{floder}/Collect_{num}/lip"

with open(face_flie, "r", encoding="utf-8") as f:
    data= json.load(f)

for path in tqdm.tqdm(data):   
    floder = path.split("\\")[1].split("/")[0]
    filename = os.path.basename(path)[:-4]
    path = f"{floder}/Collect_{num}/script/{floder}/video/{filename}mp4"
    if not os.path.exists(f"{floder}/Collect_{num}/lip/video-fpsfree-new/{floder}"):
        os.makedirs(f"{floder}/Collect_{num}/lip/video-fpsfree-new/{floder}")
    if not  os.path.exists(f"{floder}/Collect_{num}/lip/lipali-new/{floder}"):
        os.makedirs(f"{floder}/Collect_{num}/lip/lipali-new/{floder}")
    output_path = f"{out_floder}/video-fpsfree-new/{floder}/{filename}mp4"

    frames = []
    shapes = []
    errors = []
    euclidean_distances = []

    cap = cv2.VideoCapture(path)
    frame_rate = int(cap.get(cv2.CAP_PROP_FPS))
    frame_number = 0
    go = False

    # make video mount only
    output = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (140, 46))

    while cap.isOpened():
        ret, frame = cap.read()
        if ret == False:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        try:
            face = detector(gray)
            landmarks = predictor(gray, face[0])
            lip_coords = [(landmarks.part(i).x, landmarks.part(i).y) for i in range(48, 68)]
            x, y, w, h = cv2.boundingRect(np.array(lip_coords))
            kmeans = KMeans(n_clusters=1).fit(lip_coords)
            center_x, center_y = kmeans.cluster_centers_[0]
            frame = resize_data(frame, x, y, w, h)
            if frame is None:
                errors.append(output_path)
                go = True
                break
        except:
            try:
                frame = resize_data(frame, x, y, w, h)
            except:
                errors.append(output_path)
                go = True
                break
        output.write(frame)
        if frame_number % frame_rate == 0:
            euclidean_distance = np.round(pairwise_distances([[center_x, center_y]], lip_coords, metric='euclidean').flatten(), 2)
            euclidean_distances.append(list(euclidean_distance))

        frame_number += 1
    cap.release()
    output.release()
    if go:
        continue
    with open(f"{out_floder}/lipali-new/{floder}/{filename}json", "w", encoding="utf-8") as f:
        json.dump(euclidean_distances, f)