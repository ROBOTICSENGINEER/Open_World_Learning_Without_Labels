import numpy as np
import h5py


evm_model_name = input("Enter EVM name : \n") 

evm_file = input("Enter EVM model address (*.hdf5) : \n") 

n = int(input("Enter number of discovered classes (*.hdf5) : \n") )



h5 = h5py.File(evm_file, 'r')

#['CoveredVectors', 'indexes', 'scale', 'shape', 'sign', 'smallScore', 'translateAmount']

sc = list()
sh = list()

for k in range(1,1001):
  sc = sc + list(h5[f'EVM-{k}']['ExtremeVectors']['scale'][()])
  sh = sh + list(h5[f'EVM-{k}']['ExtremeVectors']['shape'][()])
np.save(file = ('known_scale_' + evm_model_name + '.npy'), arr=np.array(sc))
np.save(file = ('known_shape_' + evm_model_name + '.npy'), arr=np.array(sh))


if n > 0:
  n = n + 1000
    sc = list()
    sh = list()
    for k in range(1001,n):
      print(k)
      sc = sc + list(h5[f'EVM-{k}']['ExtremeVectors']['scale'][()])
      sh = sh + list(h5[f'EVM-{k}']['ExtremeVectors']['shape'][()])
    np.save(file = ('unknown_scale_' + evm_model_name + '.npy'), arr=np.array(sc))
    np.save(file = ('unknown_shape_' + evm_model_name + '.npy'), arr=np.array(sh))


