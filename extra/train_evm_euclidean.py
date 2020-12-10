import numpy as np
import MultipleEVM
import h5py
import torch
import time


data_path = '/scratch/feature_train_efficientnet_b3_centerloss_imagenet.npy'
output_path = '/scratch/EVM_euclidean_model_imagenet_b3_tail40000_ct7_dm45.hdf5'
N_classes = 1000


tailsize=40000
cover_threshold=0.7
distance_multiplier=0.45
distance_function='euclidean'
device='cuda'

startTime = time.time()

#Load features and labels
known = np.load(data_path,allow_pickle=True)
known = known[~np.isnan(known).any(axis=1)]
labels = []
features = []
for i in range(len(known)):
    labels.append(int(known[i][0]))
    features.append(known[i][1:])

classes = [[] for i in range(N_classes)]

print('\nAppending features to their respective class.')
#Get features of every class
for i,j in enumerate(labels):
    classes[j].append(features[i])
    
for class_num in range(len(classes)):
    classes[class_num] = torch.from_numpy(np.stack(classes[class_num]))

print('\nTime taken to load/prepare data *****************')
print(time.time() - startTime)
    
print('\nCreating an object of the EVM class & training the model.')
#Create an object of the EVM class, tailsize is the same as EVM
mevm = MultipleEVM.MultipleEVM(tailsize=tailsize, cover_threshold=cover_threshold, distance_multiplier=distance_multiplier, distance_function=distance_function, device=device)

#Train the model
mevm.train(classes,labels = list(range(0,len(classes))))

print('\nTime taken to train *****************')
print(time.time() - startTime)

print('\nSaving the model.')
#Save the model (uncomment whichever model to save)
mevm.save(output_path)

print('\nOverall runtime *****************')
print(time.time() - startTime)
