import os
import json
import time
import numpy as np
import pandas as pd
import h5py
from B3 import calc_b3


levels = ['u10', 'u25', 'u50', 'u100']

Number_of_tests = 5
number_of_batch = 50
N = 1000
number_of_window = len(range(0,2501-N, 100))


csv_root = input("Enter address of csv (input) to process : \n") 
output_h5 = input("Enter output file name (*.hdf5) : \n") 

if csv_root[-1] != '/':
  csv_root = csv_root + '/'



print(" ")

with open('./data/folder_to_id_dict_known.json', 'r') as f:
  folder_to_id_dict_known = json.load(f)
  
with open('./data/folder_to_id_dict_unknown_166.json', 'r') as f:
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


with h5py.File(output_h5, "w") as h5:
  for level in levels:
    for test_id in range(Number_of_tests): 
      
      
      
      score_post  = np.zeros(number_of_window)
      score_pre = np.zeros(number_of_window)
  
      all_L = np.array([])
      all_C = np.array([])
  
      csv_files_path = csv_root + "characterization_" + level + "_"  + str(test_id) + "_"
      charcterization_csv_fils_pre  = [csv_files_path  + "pre_"  + str(k).zfill(2) + ".csv" for k in range(number_of_batch)]
  
      for index,csv_file in enumerate(charcterization_csv_fils_pre):
        df = pd.read_csv(csv_file, sep=',',header=None, index_col=False)
        id = df.iloc[:,0]
        c = df.values[:,1:]
        L = label_finder(id)
        C = np.argmax(c, axis =1)
        is_known = (L>0) * (L<1001)
        all_L = np.append(all_L, L[is_known])
        all_C = np.append(all_C, C[is_known])
      
      
      if len(all_C)<2500:
        n = 2500 - len(all_C)
        print(f"Warning: level {level} \ttest_id {test_id} pre \t\tN = {len(all_C)}  < 2500 \tN - 2500 = {n}")
        tmp_C = np.ones(2500) * all_C[0]
        tmp_L = np.ones(2500) * all_L[0]
        tmp_C[n:] = all_C
        tmp_L[n:] = all_L
        all_C = np.copy(tmp_C)
        all_L = np.copy(tmp_L)  

        
  
      i = -1
      for k in range(0,2501-N, 100):
        i = i + 1
        n1 = k 
        n2 = k + N
        window_L = all_L[n1:n2]
        window_C = all_C[n1:n2]
        
        
        is_known_window = (window_L>0) * (window_L<1001)
        is_unknown_window = ~is_known_window
        predicted_known_window = (window_C>0) * (window_C<1001)
        predicted_unknown_window = ~predicted_known_window
        
      
        N_KK =  np.sum(is_known_window*predicted_known_window)
        N_KU =  np.sum(is_known_window*predicted_unknown_window)
        N_UK =  np.sum(is_unknown_window*predicted_known_window)
        N_UU =  np.sum(is_unknown_window*predicted_unknown_window)
      
        N_window = N_KK + N_KU + N_UK + N_UU
        assert N_window == N
      
        window_y = window_L[is_known_window*predicted_known_window]
        window_k = window_C[is_known_window*predicted_known_window]
        window_LUU = window_L[is_unknown_window*predicted_unknown_window]
        window_CUU = window_C[is_unknown_window*predicted_unknown_window]
      
        if N_KK > 0:
          correct_window = np.sum(window_y==window_k)
        else:
          correct_window = 0
        if N_UU > 0:
          b3_window, _, _ = calc_b3(L = window_LUU , K = window_CUU)
        else:
          b3_window = 0 
        
        score_pre[i] = ( correct_window +  ( b3_window * N_UU ) ) /  N
      
      h5.create_dataset(f'{level}/{test_id}/score_pre', data = score_pre , dtype=np.float64)

        
        
