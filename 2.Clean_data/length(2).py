"""
save length table are contain
-base video ex.video-2560
-filename
-time are play in video
"""

import pandas as pd
import numpy as np
import cv2
import glob
import os
import tqdm

from videotime import duration

num = 2
floder = "../data"
relation = f"{floder}/Relation.csv" #path of full relation table
length_table = f'{floder}/Collect_{num}/length.csv' # path of save length table

all_meta = {"a":1,"b":2,"c":3,"d":4,"e":5,
            "f":6,"g":7,"h":8,"i":9,"j":10}
meta_data = ["a","b","c","d","e","f","g","h","i","j"]

Collect = pd.read_csv(relation) 
Collect = Collect[Collect["index"].apply(lambda x: x[0]) == meta_data[num-1]]

b_video = [] # base video ex. video-2502
videos = [] # filename of video
durations = []
for ind in tqdm.tqdm(Collect["index"]):
    data = f'{floder}/Collect_{num}'
    scripts = f'{data}/script/video-{str(int(ind[2:]))}'
    for script in glob.glob(f"{scripts}/video/*"):
        b_video.append(os.path.basename(scripts))
        videos.append(os.path.basename(script))
        durations.append(duration(script))

dura_data = pd.DataFrame({
    "path": b_video,
    "video": videos,
    "durations": durations 
})

dura_data = dura_data.drop_duplicates(subset= ["video", "durations"], keep='first')

dura_data.to_csv(length_table, index=False)