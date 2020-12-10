import random
import json
import os
import time
import numpy as np
import pandas as pd
import h5py


from B3 import calc_b3



number_of_batch = 50
number_of_known_batch = int(number_of_batch/2)
Number_of_tests = 5

levels = ['u10', 'u25', 'u50', 'u100']


csv_root = input("Enter address of csv (input) to process : \n") 
output_h5 = input("Enter output file name (*.hdf5) : \n") 

if csv_root[-1] != '/':
  csv_root = csv_root + '/'



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




with h5py.File(output_h5, "w") as h5:
  for level in levels:
    for test_id in range(Number_of_tests): 
      
      score_diff  = np.zeros(number_of_known_batch)
      score_post  = np.zeros(number_of_known_batch)
      score_pre = np.zeros(number_of_known_batch)
    
      csv_files_path = csv_root + "characterization_" + level + "_"  + str(test_id) + "_"
      charcterization_csv_fils_pre  = [csv_files_path  + "pre_"  + str(k).zfill(2) + ".csv" for k in range(number_of_batch)]


      all_known_L = np.array([])
      all_known_C = np.array([])
      
      for index,csv_file in enumerate(charcterization_csv_fils_pre):
        df = pd.read_csv(csv_file, sep=',',header=None, index_col=False)
        id = df.iloc[:,0]
        c = df.values[:,1:]
        L = label_finder(id)
        C = np.argmax(c, axis =1)
        is_known = (L>0) * (L<1001)
        all_known_L = np.append(all_known_L, L[is_known])
        all_known_C = np.append(all_known_C, C[is_known])
      

      N = len(all_known_C)//100
      print("N = ", N)
      for n in range(N):
        i1 = n * 100
        i2 = (n+1) * 100
        L = all_known_L[i1:i2]
        C = all_known_C[i1:i2]
      
        is_known = (L>0) * (L<1001)
        is_unknown = ~is_known
        predicted_known = (C>0) * (C<1001)
        predicted_unknown = ~predicted_known
        
        
        N_KK =  np.sum(is_known*predicted_known)
        N_KU =  np.sum(is_known*predicted_unknown)
        N_UK =  np.sum(is_unknown*predicted_known)
        N_UU =  np.sum(is_unknown*predicted_unknown)
        
        N_ALL = N_KK + N_KU + N_UK + N_UU

    
        LKK = L[is_known*predicted_known]
        CKK = C[is_known*predicted_known]
        LUU = L[is_unknown*predicted_unknown]
        CUU = C[is_unknown*predicted_unknown]
    
        if N_KK > 0:
          correct = np.sum(LKK==CKK)
        else:
          correct = 0
        if N_UU > 0:
          b3, _, _ = calc_b3(L = LUU , K = CUU)
        else:
          b3 = 0 
    
        score_pre[n] = ( correct +  ( b3 * N_UU ) ) /  N_ALL
  
      if len(all_known_C) > N * 100:
        print("all_known_C.shape = ", all_known_C.shape)
        print("N changed to ", N+1)
        i1 = N * 100
        L = all_known_L[i1:]
        C = all_known_C[i1:]
      
        is_known = (L>0) * (L<1001)
        is_unknown = ~is_known
        predicted_known = (C>0) * (C<1001)
        predicted_unknown = ~predicted_known
        
        
        N_KK =  np.sum(is_known*predicted_known)
        N_KU =  np.sum(is_known*predicted_unknown)
        N_UK =  np.sum(is_unknown*predicted_known)
        N_UU =  np.sum(is_unknown*predicted_unknown)
        
        N_ALL = N_KK + N_KU + N_UK + N_UU

    
        LKK = L[is_known*predicted_known]
        CKK = C[is_known*predicted_known]
        LUU = L[is_unknown*predicted_unknown]
        CUU = C[is_unknown*predicted_unknown]
    
        if N_KK > 0:
          correct = np.sum(LKK==CKK)
        else:
          correct = 0
        if N_UU > 0:
          b3, _, _ = calc_b3(L = LUU , K = CUU)
        else:
          b3 = 0 
        score_post[n] = ( correct +  ( b3 * N_UU ) ) /  N_ALL
        
      score_diff = score_post - score_pre
      
      h5.create_dataset(f'{level}/{test_id}/score_pre', data = score_pre , dtype=np.float64)
      h5.create_dataset(f'{level}/{test_id}/score_post', data = score_post , dtype=np.float64)
      h5.create_dataset(f'{level}/{test_id}/score_diff', data = score_diff , dtype=np.float64)
      
