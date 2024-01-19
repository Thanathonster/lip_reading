import glob
import json
import tqdm
import os

num = 1
lsdata = []
floder = "../data"
collect = f"{floder}/Collect_{num}/face" # floder face
jsopfile = f"{floder}/Collect_{num}/face_1.json" # path of jsonfile data are pass condition 

for v in tqdm.tqdm(glob.glob(f"{collect}/*")):
    video = os.path.basename(v)
    aliface = f"{v}/aliface"
    for f in glob.glob(f"{aliface}/*"):
        with open(f, "r", encoding='utf8') as outfile:
            data = json.load(outfile)

        #check data are aliment are not upper than 45 and lower than -45
        yall = all([True if x and -45<x<45 else False for x in data["yall"]])
        patch = all([True if x and -45<x<45 else False for x in data["patch"]])

        if set(data["people"]) == {1} and yall and patch:
            with open(f"/Collect_{num}/face_1.json", "r", encoding = 'utf8') as load:
                lsdata = json.load(load)

            lsdata.append(f)

            with open(f"/Collect_{num}/face_1.json", "w", encoding = 'utf8') as filename:
                json.dump(lsdata, filename)