"""
merge relation 1-10 are collecting
- drop id_youtube are dupecate data
"""
import pandas as pd

meta_data = ["a","b","c","d","e","f","g","h","i","j"]
floder = ".." #path floder of Collect data
all_relation = "../relation.csv" #path want to save csv


full_data = pd.DataFrame()
for i in range(1,11):
    data = pd.read_csv(f"{floder}/Collect_{i}/Relation.csv")
    data["index"] = data["index"].apply(lambda x: f"{meta_data[i-1]}_{str(int(x)+1)}") # error in collect Youtube
    full_data = pd.concat([full_data,data], ignore_index=True)

full_data = full_data.drop_duplicates(subset= ["id"], keep='first')
full_data.to_csv(all_relation, index=False)