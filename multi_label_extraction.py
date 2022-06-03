import pandas as pd 
import re
import numpy as np
import os
import shutil

df_reports = pd.read_csv ('./indiana_reports.csv')
df_projections = pd.read_csv('./indiana_projections.csv')




df_merged = pd.merge(df_projections, df_reports, on="uid", how="left")



df_merged = df_merged[['filename', 'Problems']]



df_merged['filename'] = df_merged['filename'].map(lambda x: x.split('.')[0].split('_')[-1])
df_merged.rename(columns={'filename':'file_id'}, inplace=True)

s = set()
for problem in df_merged['Problems']:
    for symptom in re.split(';' ,problem):
        s.add(symptom)

blacklist = ["Pulmonary Artery", "Stents", "Lung", "Heart", "Sulcus", "Heart Ventricles", 'Bone and Bones',
               "Sutures", "Surgical Instruments", "Tube, Inserted",
               "Thorax", "Contrast Media", "Blood Vessels", "No Indexing", "Cervical Vertebrae",
               "Mediastinum", "Abdomen", "Medical Device", "Shift","Thickening",
               "Trachea", "Pleura", "Lymph nodes", "Technical Quality of Image Unsatisfactory","Mastectomy",
                "Ribs", "Cavitation", "Lucency", "Spine", "Aorta",
                "Lumbar Vertebrae", "Thoracic vertebrae", "Catheters, Indwelling",
                "Breast Implants", "Volume Loss", "Implanted Medical Device",
                "Markings", "Lymph Nodes", "Diaphragm", "Opacity", "Trachea, Carina", "Adipose Tissue" , "Density",
                "Nipple Shadow","Humerus", "Heart Atria", "Costophrenic Angle", "Shoulder"]

print(f"unique keywords\t\t{len(s)}")
for elem in blacklist:
    s.discard(elem)
print(f"relevant keywords\t{len(s)}")


occurences = {}
for problems in df_merged['Problems']:
    for problem in s:
        if(problem in problems):
            if(problem in occurences):
                occurences[problem] += 1
            else:
                occurences[problem] = 1
occurences = dict(sorted(occurences.items(), key=lambda item: -item[1]))


cut_off = 7
prominent_classes = []
other_classes = []
print("top 7 labels in terms of occurences:")
for key, value in occurences.items():
    if(cut_off>0):
        print(f"\t{key}({value})")
        prominent_classes.append(key)
        cut_off -= 1
    else:
        other_classes.append(key)
    
import math

def get_label(row, nb_occ):
    min_occ = math.inf
    choosen_one = 0
    for index, elem in enumerate(row):
        if(elem and nb_occ[index] < min_occ):
            choosen_one = index
            min_occ = nb_occ[index]
    nb_occ[choosen_one] += 1
    return choosen_one


shutil.rmtree('./image_labels_csv_new')
os.mkdir('./image_labels_csv_new')
multiclass=False
for csv_file in os.listdir("./image_labels_csv"):
    
    df_current = pd.read_csv("./image_labels_csv/" + csv_file)
    df_current.rename(columns={'Label': 'label'}, inplace=True)
    nb_occ = {}
    for index in range(8):
        nb_occ[index] = 0
    df_file_names = pd.concat([df_current['Filename'], df_current['Filename'].map(lambda x: x.split('.')[0].split('_')[-1]).rename('file_id')], axis=1)
    df_join = pd.merge(df_file_names, df_merged, on='file_id', how='left')
    df_join['Problems'] = df_join['Problems'].fillna('other')
    for index, label in enumerate(prominent_classes):
        df_join[label] = df_join['Problems'].map(lambda problems:index if label in problems else 0)
    df_join["other"] = df_join['Problems'].map(lambda problems: 1 if any(x in problems for x in other_classes) else 0)
    df_current = pd.merge(df_current, df_join, on='Filename', how='left')
    df_current = df_current.drop(['file_id', 'Problems'], axis=1)
    df_current = df_current.drop_duplicates()
    df_current = df_current.groupby(['Filename', 'LabelText'], as_index=False).sum()
    for label in prominent_classes:
        df_current[label] = df_current[label].map(lambda x: 1 if x>=1 else 0)
    if(multiclass):
        df_current['label'] = df_current[prominent_classes + ['other']].apply(lambda row: get_label(row, nb_occ), axis=1)
        df_current = df_current[['Filename', 'label', 'LabelText']]
    else:
        df_current = df_current[['Filename'] + prominent_classes + ['other', 'LabelText']]
    df_current.to_csv("./image_labels_csv_new/" + csv_file, index=False)






