import os
import json
import time
import numpy as np
import pandas as pd
from  sklearn.metrics import normalized_mutual_info_score
from B3 import calc_b3


levels = ['easy', 'medium', 'hard', 'extreme']
Number_of_tests = 5
number_of_batch = 50
N = 1000

#csv_root = './csv_folder/light_045_065/'
#csv_root = './csv_folder/mocoImagenet_045_065/'
#csv_root = './csv_folder/mocoPlaces_045_065/'
#csv_root = './csv_folder/mocoImagenetPlaces_045_065/'
#csv_root = './csv_folder/supervised_mocoImagenet_045_065/'
#csv_root = './csv_folder/supervised_mocoPlaces_045_065/'
#csv_root = './csv_folder/supervised_mocoImagenetPlaces_045_065/'
#csv_root = './csv_folder/Label_045_065/'
#csv_root = './csv_folder/label_mocoImagenet_045_065/'
csv_root = './csv_folder/label_mocoPlaces_045_065/'

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
  
  acc_pre = np.zeros(Number_of_tests)
  nmi_pre = np.zeros(Number_of_tests)
  b3_pre = np.zeros(Number_of_tests)  
  acc_known_pre = np.zeros(Number_of_tests)
  nmi_known_pre = np.zeros(Number_of_tests)
  b3_known_pre = np.zeros(Number_of_tests)  
  acc_unknown_pre = np.zeros(Number_of_tests)
  nmi_unknown_pre = np.zeros(Number_of_tests)
  b3_unknown_pre = np.zeros(Number_of_tests)
  acc_post = np.zeros(Number_of_tests)
  nmi_post = np.zeros(Number_of_tests)
  b3_post = np.zeros(Number_of_tests)  
  acc_known_post = np.zeros(Number_of_tests)
  nmi_known_post = np.zeros(Number_of_tests)
  b3_known_post = np.zeros(Number_of_tests)  
  acc_unknown_post = np.zeros(Number_of_tests)
  nmi_unknown_post = np.zeros(Number_of_tests)
  b3_unknown_post = np.zeros(Number_of_tests)
  
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
    last_y_pre = np.copy(last_L_pre)
    last_y_pre[last_y_pre>1000] = 0
    last_k_pre = np.copy(last_C_pre)
    last_k_pre[last_k_pre>1000] = 0
    is_known_pre = (last_L_pre>0) * (last_L_pre<1001)
    is_unknown_pre = ~is_known_pre
    last_known_L_pre = last_L_pre[is_known_pre]
    last_known_C_pre = last_C_pre[is_known_pre]
    last_known_y_pre = last_y_pre[is_known_pre]
    last_known_k_pre = last_k_pre[is_known_pre]
    last_unknown_L_pre = last_L_pre[is_unknown_pre]
    last_unknown_C_pre = last_C_pre[is_unknown_pre]
    last_unknown_y_pre = last_y_pre[is_unknown_pre]
    last_unknown_k_pre = last_k_pre[is_unknown_pre]
    
    acc_pre[test_id] = np.sum(last_y_pre==last_k_pre)/len(last_y_pre)
    nmi_pre[test_id] = normalized_mutual_info_score(labels_true = last_L_pre, labels_pred=last_C_pre, average_method='max')
    b3_pre[test_id], _, _ = calc_b3(L = last_L_pre , K = last_C_pre) 
    acc_known_pre[test_id] = np.sum(last_known_y_pre==last_known_k_pre)/len(last_known_y_pre)
    nmi_known_pre[test_id] = normalized_mutual_info_score(labels_true = last_known_L_pre, labels_pred=last_known_C_pre, average_method='max')
    b3_known_pre[test_id], _, _ = calc_b3(L = last_known_L_pre , K = last_known_C_pre) 
    acc_unknown_pre[test_id] = np.sum(last_unknown_y_pre==last_unknown_k_pre)/len(last_unknown_y_pre)
    nmi_unknown_pre[test_id] = normalized_mutual_info_score(labels_true = last_unknown_L_pre, labels_pred=last_unknown_C_pre, average_method='max')
    b3_unknown_pre[test_id], _, _ = calc_b3(L = last_unknown_L_pre , K = last_unknown_C_pre) 



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
    last_y_post = np.copy(last_L_post)
    last_y_post[last_y_post>1000] = 0
    last_k_post = np.copy(last_C_post)
    last_k_post[last_k_post>1000] = 0
    is_known_post = (last_L_post>0) * (last_L_post<1001)
    is_unknown_post = ~is_known_post
    last_known_L_post = last_L_post[is_known_post]
    last_known_C_post = last_C_post[is_known_post]
    last_known_y_post = last_y_post[is_known_post]
    last_known_k_post = last_k_post[is_known_post]
    last_unknown_L_post = last_L_post[is_unknown_post]
    last_unknown_C_post = last_C_post[is_unknown_post]
    last_unknown_y_post = last_y_post[is_unknown_post]
    last_unknown_k_post = last_k_post[is_unknown_post]
    
    acc_post[test_id] = np.sum(last_y_post==last_k_post)/len(last_y_post)
    nmi_post[test_id] = normalized_mutual_info_score(labels_true = last_L_post, labels_pred=last_C_post, average_method='max')
    b3_post[test_id], _, _ = calc_b3(L = last_L_post , K = last_C_post) 
    acc_known_post[test_id] = np.sum(last_known_y_post==last_known_k_post)/len(last_known_y_post)
    nmi_known_post[test_id] = normalized_mutual_info_score(labels_true = last_known_L_post, labels_pred=last_known_C_post, average_method='max')
    b3_known_post[test_id], _, _ = calc_b3(L = last_known_L_post , K = last_known_C_post) 
    acc_unknown_post[test_id] = np.sum(last_unknown_y_post==last_unknown_k_post)/len(last_unknown_y_post)
    nmi_unknown_post[test_id] = normalized_mutual_info_score(labels_true = last_unknown_L_post, labels_pred=last_unknown_C_post, average_method='max')
    b3_unknown_post[test_id], _, _ = calc_b3(L = last_unknown_L_post , K = last_unknown_C_post) 
    
  
  #print(f"\n{level = }")  
  
  '''
  print("pre known accuracy = ",  np.around(acc_known_pre, 4))
  print("pre known NMI = ",  np.around(nmi_known_pre, 4))
  print("pre known B3 = ", np.around(b3_known_pre, 4) )

  print("pre unknown accuracy = ",  np.around(acc_unknown_pre, 4))
  print("pre unknown NMI = ",  np.around(nmi_unknown_pre, 4))
  print("pre unknown B3 = ", np.around(b3_unknown_pre, 4) )

  print("pre mixed accuracy = ",  np.around(acc_pre, 4))
  print("pre mixed NMI = ",  np.around(nmi_pre, 4))
  print("pre mixed B3 = ", np.around(b3_pre, 4) )

  print("post known accuracy = ",  np.around(acc_known_post, 4))
  print("post known NMI = ",  np.around(nmi_known_post, 4))
  print("post known B3 = ", np.around(b3_known_post, 4) )

  print("post unknown accuracy = ",  np.around(acc_unknown_post, 4))
  print("post unknown NMI = ",  np.around(nmi_unknown_post, 4))
  print("post unknown B3 = ", np.around(b3_unknown_post, 4) )

  print("post mixed accuracy = ",  np.around(acc_post, 4))
  print("post mixed NMI = ",  np.around(nmi_post, 4))
  print("post mixed B3 = ", np.around(b3_post, 4) )
  '''
  '''
  print("pre known accuracy = ",  round( np.mean(acc_known_pre) , 4))
  print("pre known NMI = ",  round( np.mean(nmi_known_pre) , 4))
  print("pre known B3 = ", round( np.mean(b3_known_pre) , 4) )

  print("pre unknown accuracy = ",  round( np.mean(acc_unknown_pre) , 4))
  print("pre unknown NMI = ",  round( np.mean(nmi_unknown_pre) , 4))
  print("pre unknown B3 = ", round( np.mean(b3_unknown_pre) , 4) )

  print("pre mixed accuracy = ",  round( np.mean(acc_pre) , 4))
  print("pre mixed NMI = ",  round( np.mean(nmi_pre) , 4))
  print("pre mixed B3 = ", round( np.mean(b3_pre) , 4) )

  print("post known accuracy = ",  round( np.mean(acc_known_post) , 4))
  print("post known NMI = ",  round( np.mean(nmi_known_post) , 4))
  print("post known B3 = ", round( np.mean(b3_known_post) , 4) )

  print("post unknown accuracy = ",  round( np.mean(acc_unknown_post) , 4))
  print("post unknown NMI = ",  round( np.mean(nmi_unknown_post) , 4))
  print("post unknown B3 = ", round( np.mean(b3_unknown_post) , 4) )

  print("post mixed accuracy = ",  round( np.mean(acc_post) , 4))
  print("post mixed NMI = ",  round( np.mean(nmi_post) , 4))
  print("post mixed B3 = ", round( np.mean(b3_post) , 4) )
  '''

  print(round( np.mean(acc_pre) , 4) , round( np.mean(acc_post) , 4) , sep=' & ' , end=' & ')
  
print("\n")
