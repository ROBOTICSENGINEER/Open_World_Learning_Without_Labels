import os
import json
import time
import numpy as np
import pandas as pd
from B3 import calc_b3

N = 1000

levels = ['u10', 'u25', 'u50', 'u100']

Number_of_tests = 5
number_of_batch = 50


csv_root = input("Enter address of csv (input) to process : \n") 

if csv_root[-1] != '/':
  csv_root = csv_root + '/'


print(" ")

with open('./preparing/folder_to_id_dict_known.json', 'r') as f:
  folder_to_id_dict_known = json.load(f)
  
with open('./preparing/folder_to_id_dict_unknown_166.json', 'r') as f:
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
  
  score_pre = np.zeros(Number_of_tests)
  score_post = np.zeros(Number_of_tests)
  
  for test_id in range(Number_of_tests): 

    all_L_pre = np.array([])
    all_C_pre = np.array([])

    csv_files_path = csv_root + "characterization_" + level + "_"  + str(test_id) + "_"
    charcterization_csv_fils_pre  = [csv_files_path  + "pre_"  + str(k).zfill(2) + ".csv" for k in range(number_of_batch)]

    for index,csv_file in enumerate(charcterization_csv_fils_pre):
      df = pd.read_csv(csv_file, sep=',',header=None, index_col=False)
      id = df.iloc[:,0]
      c = df.values[:,1:]
      L = label_finder(id)
      C = np.argmax(c, axis =1)
      all_L_pre = np.append(all_L_pre, L)
      all_C_pre = np.append(all_C_pre, C)
      
    last_L_pre = all_L_pre[-N:]
    last_C_pre = all_C_pre[-N:]
    
    is_known_pre = (last_L_pre>0) * (last_L_pre<1001)
    is_unknown_pre = ~is_known_pre
    predicted_known_pre = (last_C_pre>0) * (last_C_pre<1001)
    predicted_unknown_pre = ~predicted_known_pre
    
    
    N_KK_pre =  np.sum(is_known_pre*predicted_known_pre)
    N_KU_pre =  np.sum(is_known_pre*predicted_unknown_pre)
    N_UK_pre =  np.sum(is_unknown_pre*predicted_known_pre)
    N_UU_pre =  np.sum(is_unknown_pre*predicted_unknown_pre)
    
    N_pre = N_KK_pre + N_KU_pre + N_UK_pre + N_UU_pre
    assert N_pre == N
    
    last_y_pre = last_L_pre[is_known_pre*predicted_known_pre]
    last_k_pre = last_C_pre[is_known_pre*predicted_known_pre]
    last_LUU_pre = last_L_pre[is_unknown_pre*predicted_unknown_pre]
    last_CUU_pre = last_C_pre[is_unknown_pre*predicted_unknown_pre]
    
    if N_KK_pre > 0:
      correct_pre = np.sum(last_y_pre==last_k_pre)
    else:
      correct_pre = 0
    if N_UU_pre > 0:
      b3_pre, _, _ = calc_b3(L = last_LUU_pre , K = last_CUU_pre)
    else:
      N_UU_pre = 0 
    
    score_pre[test_id] = ( correct_pre +  ( b3_pre * N_UU_pre ) ) /  N_pre
    
    
    all_L_post = np.array([])
    all_C_post = np.array([])

    csv_files_path = csv_root + "characterization_" + level + "_"  + str(test_id) + "_"
    charcterization_csv_fils_post  = [csv_files_path  + "post_"  + str(k).zfill(2) + ".csv" for k in range(number_of_batch)]

    for index,csv_file in enumerate(charcterization_csv_fils_post):
      df = pd.read_csv(csv_file, sep=',',header=None, index_col=False)
      id = df.iloc[:,0]
      c = df.values[:,1:]
      L = label_finder(id)
      C = np.argmax(c, axis =1)
      all_L_post = np.append(all_L_post, L)
      all_C_post = np.append(all_C_post, C)
      
    last_L_post = all_L_post[-N:]
    last_C_post = all_C_post[-N:]
    
    is_known_post = (last_L_post>0) * (last_L_post<1001)
    is_unknown_post = ~is_known_post
    predicted_known_post = (last_C_post>0) * (last_C_post<1001)
    predicted_unknown_post = ~predicted_known_post
    
    
    N_KK_post =  np.sum(is_known_post*predicted_known_post)
    N_KU_post =  np.sum(is_known_post*predicted_unknown_post)
    N_UK_post =  np.sum(is_unknown_post*predicted_known_post)
    N_UU_post =  np.sum(is_unknown_post*predicted_unknown_post)
    
    N_post = N_KK_post + N_KU_post + N_UK_post + N_UU_post
    assert N_post == N
    
    last_y_post = last_L_post[is_known_post*predicted_known_post]
    last_k_post = last_C_post[is_known_post*predicted_known_post]
    last_LUU_post = last_L_post[is_unknown_post*predicted_unknown_post]
    last_CUU_post = last_C_post[is_unknown_post*predicted_unknown_post]
    
    if N_KK_post > 0:
      correct_post = np.sum(last_y_post==last_k_post)
    else:
      correct_post = 0
    if N_UU_post > 0:
      b3_post, _, _ = calc_b3(L = last_LUU_post , K = last_CUU_post)
    else:
      b3_post = 0 
    
    score_post[test_id] = ( correct_post +  ( b3_post * N_UU_post ) ) /  N_post
    

  print(round( np.mean(score_pre) , 4) , round( np.mean(score_post) , 4) , sep=' & ' , end=' & ')
  
print("\n")
