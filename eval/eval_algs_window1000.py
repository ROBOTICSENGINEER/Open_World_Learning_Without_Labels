import time
import os
import numpy as np
import pandas as pd
import json
from B3 import calc_b3
import argparse


parser = argparse.ArgumentParser(description='Evaluation of Open World Learning')
parser.add_argument('--input_csv', required=True, help='Path to csv result to evaluate it.')
parser.add_argument('--output_txt', required=True, help='txt file name to save the result.')
args_eval = parser.parse_args()

assert os.path.isfile(args_eval.input_csv) 


csv_path = args_eval.input_csv
output_txt = args_eval.output_txt


folder_to_number_dict = json.load(open("./data/folder_to_id_dict_all.json"))



def get_label(x):
  z_train = max(x.find('v2_73/') ,  x.find('train/'))
  z_val = x.find('val_in_folders/')
  
  if z_train>=0:
    ind = int(   folder_to_number_dict [     x[(z_train+6):(z_train+15)]   ]  )
  elif z_val>=0:
    ind = int(   folder_to_number_dict [     x[(z_val+15):(z_val+24)]   ]  )
  else:
    raise ValueError(f"x is not in folder_to_number_dict")
  return ind


df = pd.read_csv(csv_path, header=None, index_col=False)
id = df.iloc[:,0]
p = df.iloc[:,1:].values
y = np.argmax(p, axis=1)


L = np.array([get_label(id[k]) for k in range(5000)])


scores = np.zeros(41)

for k in range(41):
  n1 = k * 100
  n2 = n1 + 1000
  
  
  predicted = y[n1:n2]
  Labels = L[n1:n2]

  is_known = (Labels < 1001)
  is_unknown = (Labels >= 1001)
  n_known = np.sum(is_known)
  n_unknown = np.sum(is_unknown)
  
  #print([k, n1, n2, n_known, n_unknown, n_known + n_unknown ])

  if n_known == 0:
    f_measure, precision, recall = calc_b3(L = Labels , K = predicted)
    scores[k] = f_measure
  elif n_unknown == 0:
    acc = np.sum( Labels == predicted ) / 1000
    scores[k] = acc
  else:
    predicted_known = (predicted < 1001) * (predicted > 0)
    predicted_unknown = ~ predicted_known
    is_known_predicted_known = is_known * predicted_known
    is_unknown_predicted_unknown = is_unknown * predicted_unknown
    n_kk = np.sum(is_known_predicted_known)
    n_uu = np.sum(is_unknown_predicted_unknown)
    score_kk = 0.0
    score_uu = 0.0
    if n_kk > 0:
      Labels_kk = Labels[is_known_predicted_known]
      predicted_kk = predicted[is_known_predicted_known]
      score_kk = np.sum( Labels_kk == predicted_kk) 
    if n_uu > 0:
      Labels_uu = Labels[is_unknown_predicted_unknown]
      predicted_uu = predicted[is_unknown_predicted_unknown]
      f_measure, precision, recall = calc_b3(L = Labels_uu , K = predicted_uu)
      score_uu = n_uu * f_measure
    scores[k] = ( score_kk + score_uu ) / 1000


np.savetxt(fname = output_txt , X = scores, fmt='%1.8f', delimiter=',')

print("\nEnd\n")
