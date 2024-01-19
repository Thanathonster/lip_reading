"""
collect file 
- Raw video ex. video-2563 (number of video are collect) have relation with meta data
- script video ** filename is a label **
"""

import numpy as np
import pandas as pd
import json
import pythainlp
from tqdm import tqdm

from id_youtube import search_youtube
from collectfile import collect_metadata, transcript, save_file_youtube, script_video

# Fill data
floder = "data/Collect_3" #folder data collect
number = 18000 # fill max data script 6000
raw = f"{floder}/Raw/"

# pythainlp keyword dataset
f = open('data.json', encoding='utf8')
data = json.load(f)

# record the last data
with open(f"{floder}/Record.json", "r", encoding='utf8') as outfile:
    json_obj= json.load(outfile)

start = int(list(json_obj)[0])
startnum = start

for num, word in enumerate(tqdm(data[start:number])): 
    with open(f"{floder}/Record.json", "w", encoding='utf8') as outfile:
        json_object = {num:word}
        json.dump(json_object, outfile, ensure_ascii=False)
    startnum += 1

    ids, channels = search_youtube(word) #search data in YouTube
    for id,channel in zip(ids, channels):
        num_video = collect_metadata(floder, id, channel) #add data in history table
        if num_video:
            vsf = f"{floder}/script/video-{num_video}"
            try:
                srt = transcript(id) #download transcript
                save_file_youtube(id, raw, num_video) #save YouTube video
                error = script_video(vsf,srt, f"{raw}/{num_video}.mp4") #cut video to script YouTube video
                print(error)
            except:
                continue
        else:
            continue