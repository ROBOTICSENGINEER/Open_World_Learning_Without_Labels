'''
test = 5000 images = 2500 knowns (50 class) + 2500 unknowns (N class)

known (ImageNet 2012, 1000 classes):
100 class x 25  image/class = 2500 images

Unkwons (ImageNet 2010, 166 classes):
u10  =  10  class x 500 image/class
u25  =  25  class x 100 image/class
u50  =  50  class x 50  image/class
u100 =  100 class x 25  image/class

Batch size  = 100
Batch/test = 50
levels = 4

tests_per_level = 100? 50? 25? 20? 10?

total tests = 2 x 4 x tests_per_level = 8 x tests_per_level

'''


import random
import numpy as np
import pandas as pd

seed = 2
random.seed(seed) 
np.random.seed(seed) 



N_test = 5  # for generating statistical results
NC_known = 100 # number of known classes
N_kown_per_class = 25

levels = ['u10' , 'u25' , 'u50', 'u100']


df_known_all = pd.read_csv('imagenet_1000_val.csv', header=None, index_col=False, names = ['Path', 'Label'])
df_unknown_all =  pd.read_csv('imagenet_166.csv', header=None, index_col=False, names = ['Path', 'Label'])


for level in levels:
  if level == 'u10':
    NC_unknown = 10
    N_unknown_per_class = 250
  elif level == 'u25':
    NC_unknown = 25
    N_unknown_per_class = 100
  elif level == 'u50':
    NC_unknown = 50
    N_unknown_per_class = 50
  elif level == 'u100':
    NC_unknown = 100
    N_unknown_per_class = 25
  else:
    raise ValueError()

  
  for test_id in range(N_test):
    loop_seed = seed + 1 + test_id
    known_class_numbers =  np.random.randint(low=1, high=1000, size=NC_known, dtype=int)
    unknown_class_numbers =  np.random.randint(low=1001, high=1166, size=NC_unknown, dtype=int)
    df_test = pd.DataFrame(columns = ['Path', 'Label'])
    for known_id in known_class_numbers:
      a = df_known_all.loc[df_known_all['Label'] == known_id]
      assert 50 == len(a.index)
      a = a.sample(n=N_kown_per_class, random_state = loop_seed)
      df_test = pd.concat([df_test,a], ignore_index=True, sort=False)
    for unknown_id in unknown_class_numbers:
      a = df_unknown_all.loc[df_unknown_all['Label'] == unknown_id]
      assert N_unknown_per_class <= len(a.index)
      a = a.sample(n=N_unknown_per_class, random_state = loop_seed+1)
      print(unknown_id, len(a.index), N_unknown_per_class)
      df_test = pd.concat([df_test,a], ignore_index=True, sort=False)
    df_test = df_test.sample(frac=1, random_state = (loop_seed+2))
    assert 5000 == len(df_test.index)
    csv_name = f'./all_test_166/test_{level}_{test_id}.csv'
    df_test.to_csv(csv_name, header=False, index=False)
  
