import os
import json
import time
import numpy as np
import pandas as pd
from  sklearn.metrics import normalized_mutual_info_score
from B3 import calc_b3


levels = ['u10', 'u25', 'u50', 'u100']

Number_of_tests = 5
number_of_batch = 50

csv_root = input("Enter address of csv (input) to process : \n") 

with open('../data/folder_to_id_dict_known.json', 'r') as f:
  folder_to_id_dict_known = json.load(f)
  
with open('../data/folder_to_id_dict_unknown_166.json', 'r') as f:
  folder_to_id_dict_unknown = json.load(f)


def label_finder(id):
  y = np.zeros(len(id))
  for (k,x) in enumerate(id):
    z = x.find('/n')
    if x.find('ILSVRC_2012')>0:
      folder_name = x[(z+1):(z+10)]
      y[k] = folder_to_id_dict_known[folder_name]
    elif x.find('ILSVRC_2010')>0:
      folder_name = x[(z+1):(z+10)]
      y[k] = folder_to_id_dict_unknown[folder_name]
    else:
      raise ValueError()
  return y



for level in levels:
  
  all_L = np.array([])
  all_C = np.array([])
  all_known_L = np.array([])
  all_known_C = np.array([])
  
  for test_id in range(Number_of_tests): 

    csv_files_path = csv_root + "characterization_" + level + "_"  + str(test_id) + "_"
    charcterization_csv_fils_pre  = [csv_files_path  + "pre_"  + str(k).zfill(2) + ".csv" for k in range(number_of_batch)]

    for index,csv_file in enumerate(charcterization_csv_fils_pre):
      df = pd.read_csv(csv_file, sep=',',header=None, index_col=False)
      id = df.iloc[:,0]
      c = df.values[:,1:]
      L = label_finder(id)
      C = np.argmax(c, axis =1)
      is_known = (L>0) * (L<1001)
      all_L = np.append(all_L, L)
      all_C = np.append(all_C, C)
      all_known_L = np.append(all_known_L, L[is_known])
      all_known_C = np.append(all_known_C, C[is_known])
  assert np.max(all_C) <= 1000
  
  all_L[all_L>1000]=0
  acc_known = np.sum(all_known_L==all_known_C) / len(all_known_L)
  acc_all =  np.sum(all_L==all_C) / len(all_L)
  NMI_known = normalized_mutual_info_score(labels_true = all_known_L, labels_pred=all_known_C, average_method='max')
  NMI_all = normalized_mutual_info_score(labels_true = all_L, labels_pred=all_C, average_method='max')
  f_measure_knonw, _, _ = calc_b3(L = all_known_L , K = all_known_C) 
  f_measure_all, _, _ = calc_b3(L = all_L , K = all_C) 
  
  print(f"\n{level = }")  
  
  print("closed-set accuracy = ",  round(acc_known, 4))
  print("closed-set NMI = ",  round(NMI_known, 4))
  print("closed-set B3 = ", round(f_measure_knonw, 4) )
  
  print("open-set accuracy = ",  round(acc_all, 4))
  print("open-set NMI = ",  round(NMI_all, 4))
  print("open-set B3 = ", round(f_measure_all, 4) )
  
