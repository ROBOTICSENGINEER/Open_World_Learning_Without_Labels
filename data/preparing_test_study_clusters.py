import numpy as np
import pandas as pd
import os
import json


"""
test 1
325 ==> 70, 81, 90, 95, 96, 99, 102, 103, 104, 160

test 2
739 ==> 45, 93, 97, 98, 100, 105, 110, 116, 156, 157

test 3
917 ==>  69, 82, 122, 124, 125, 126, 127, 131, 136, 137

test 4
924 ==>  2, 3, 4, 5, 24, 39, 40, 117, 130, 166

test 5
990 ==> 49, 50, 51, 52, 54, 146, 149, 150, 153, 154
"""

"""

with open('new_166_dict.json', 'r') as f:
    unknown_166_dict = json.load(f)

unknown_166_inv_dict = {}
for k, v in unknown_166_dict.items():
  unknown_166_inv_dict[v] = k


test_1 = [1070, 1081, 1090, 1095, 1096, 1099, 1102, 1103, 1104, 1160]
test_2 = [1045, 1093, 1097, 1098, 1100, 1105, 1110, 1116, 1156, 1157]
test_3 = [1069, 1082, 1122, 1124, 1125, 1126, 1127, 1131, 1136, 1137]
test_4 = [1002, 1003, 1004, 1005, 1024, 1039, 1040, 1117, 1130, 1166]
test_5 = [1049, 1050, 1051, 1052, 1054, 1146, 1149, 1150, 1153, 1154]

test_list = [test_1 , test_2, test_3, test_4, test_5]

path_166 = '/scratch/datasets/ImageNet/ILSVRC_2010-360classes/train/'
for k, test in enumerate(test_list):
  with open(f"selected_imagenet_all_unknown_{k+1}.csv", 'w') as csv_file:
    for class_number in test:
      folder_name = unknown_166_inv_dict[str(class_number)]
      folder_path = path_166 + folder_name
      filenames= os.listdir (folder_path)
      for filename in filenames:
        full_path = folder_path + '/' + filename
        csv_file.write(full_path + ',' + str(class_number) + '\n')
  
"""


# ["imagenet_test_v2_73.csv", "imagenet_test_v2_85.csv" , "imagenet_test_v2_93.csv"]


df_known = pd.read_csv("imagenet_test_v2_93.csv", header=None)
for k in range(5):
    df_unknown = pd.read_csv(f"selected_imagenet_all_unknown_{k+1}.csv", header=None)
    csv_file_output = f"test_study_clusters_easy_{k+1}.csv"
    df_known = df_known.sample(frac=1)
    df_unknown = df_unknown.sample(frac=1)
    n_all = 5000
    n_unknown = 2500
    n_known = n_all - n_unknown
    d_known = df_known.iloc[:n_known, :]
    d_unknown = df_unknown.iloc[:n_unknown, :]
    d_concat = pd.concat([d_known, d_unknown])
    d_test = d_concat.sample(frac=1)
    d_test.to_csv(csv_file_output, index=False, header=False)


df_known = pd.read_csv("imagenet_test_v2_73.csv", header=None)
for k in range(5):
    df_unknown = pd.read_csv(f"selected_imagenet_all_unknown_{k+1}.csv", header=None)
    csv_file_output = f"test_study_clusters_hard_{k+1}.csv"
    df_known = df_known.sample(frac=1)
    df_unknown = df_unknown.sample(frac=1)
    n_all = 5000
    n_unknown = 2500
    n_known = n_all - n_unknown
    d_known = df_known.iloc[:n_known, :]
    d_unknown = df_unknown.iloc[:n_unknown, :]
    d_concat = pd.concat([d_known, d_unknown])
    d_test = d_concat.sample(frac=1)
    d_test.to_csv(csv_file_output, index=False, header=False)
