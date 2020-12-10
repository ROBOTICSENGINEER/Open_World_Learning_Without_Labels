import numpy as np
import os
import json

folder_to_id_dict = dict()
with open("imagenet_folder_names.csv", 'r') as f:
  for line in f:
    if line != '':
      print(line)
      folder, label, description = line.split(',')
      folder_to_id_dict[folder] = label


with open('folder_to_id_dict_known.json', 'w') as f:
    json.dump(folder_to_id_dict, f)


path_train = '/scratch/datasets/ImageNet/ILSVRC_2012/train/'


with open("imagenet_1000_train.csv", 'w') as f:
  for folder,class_number in folder_to_id_dict.items():
    folder_path = path_train + folder
    filenames= os.listdir (folder_path)
    for filename in filenames:
      full_path = folder_path + '/' + filename
      f.write(full_path + ',' + class_number + '\n')


path_val = '/scratch/datasets/ImageNet/ILSVRC_2012/val_in_folders/'


with open("imagenet_1000_val.csv", 'w') as f:
  for folder,class_number in folder_to_id_dict.items():
    folder_path = path_val + folder
    filenames= os.listdir (folder_path)
    for filename in filenames:
      full_path = folder_path + '/' + filename
      f.write(full_path + ',' + class_number + '\n')



with open('new_166_dict.json', 'r') as f:
    new_179_dict = json.load(f)

path_166 = '/scratch/datasets/ImageNet/ILSVRC_2010-360classes/train/'


with open("imagenet_166.csv", 'w') as f:
  for folder,class_number in new_179_dict.items():
    folder_path = path_166 + folder
    filenames= os.listdir (folder_path)
    for filename in filenames:
      full_path = folder_path + '/' + filename
      f.write(full_path + ',' + class_number + '\n')
