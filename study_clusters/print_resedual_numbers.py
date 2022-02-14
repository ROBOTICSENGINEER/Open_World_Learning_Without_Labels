import numpy as np
import cv2
import os

folder = "/scratch/mjafarzadeh/result_5_svm/"

N_k = []
N_u = []
N_a = []

for level in ["easy" , "hard"]:
  for test_id in range(1,6):
    #print(f"test {level} {test_id}")
    file = folder + f"track_svm_{level}_{test_id}_residual_dict_50.txt"
    
    nk = 0
    nu = 0
    
    with open(file, 'r') as f:
      for L in f:
        line = L.strip('\n')
        if len(line) > 5:
          if "ILSVRC_2010" in line:
            nu = nu + 1
          else:
            nk = nk + 1
    N_k.append(nk)
    N_u.append(nu)
    N_a.append(nk+nk)


i = 0
print("residual: [unknown , known , all] ")
for level in ["easy" , "hard"]:
  for test_id in range(1,6):
    print(f"{level}_{test_id} , {N_u[i]} , {N_k[i]} , {N_a[i]}")
    i = i + 1

    
