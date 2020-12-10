import numpy as np
import ..MultipleEVM
import h5py
import torch
import time

data_path = input("Enter address of features (*.npy) to train EVM : \n") 

N_classes = 1000
tailsize=33998
cover_threshold=0.7
distance_multiplier=0.45
distance_function='cosine'

print("\n\n\nPlease enter number corresponding to the features to train EVM:")
print("1 ==> supervised")
print("2 ==> moco imagenet")
print("3 ==> moco places2")
print("4 ==> moco imagenet + moco places2")
print("5 ==> supervised + moco imagenet")
print("6 ==> supervised +  moco places2")
print("7 ==> supervised + moco imagenet + moco places2\n\n")
a = input()
print("\n\n\nyou have entered ", a)

if a == '1':
  print("FV = [ supervised ]")
  output_path = f"/scratch/EVM_{distance_function}_model_imagenet_b3_supervised_tail{tailsize}_ct{100*cover_threshold}_dm{100*distance_multiplier}.hdf5"
elif a == '2':
  print("FV = [ moco imagenet ]")
  output_path = f"/scratch/EVM_{distance_function}_model_imagenet_b3_mocoImagenet_tail{tailsize}_ct{100*cover_threshold}_dm{100*distance_multiplier}.hdf5"
elif a == '3':
  print("FV = [ moco places2 ]")
  output_path = f"/scratch/EVM_{distance_function}_model_imagenet_b3_mocoPlaces_tail{tailsize}_ct{100*cover_threshold}_dm{100*distance_multiplier}.hdf5"
elif a == '4':
  print("FV = [ moco imagenet + moco places2 ]")
  output_path = f"/scratch/EVM_{distance_function}_model_imagenet_b3_mocoImagenetPlaces2_tail{tailsize}_ct{100*cover_threshold}_dm{100*distance_multiplier}.hdf5"
elif a == '5':
  print("FV = [ supervised , moco imagenet ]")
  output_path = f"/scratch/EVM_{distance_function}_model_imagenet_b3_joint_supervised_mocoImagenet_tail{tailsize}_ct{100*cover_threshold}_dm{100*distance_multiplier}.hdf5"
elif a == '6':
  print("FV = [ supervised ,  moco places2 ]")
  output_path = f"/scratch/EVM_{distance_function}_model_imagenet_b3_joint_supervised_mocoPlaces2_tail{tailsize}_ct{100*cover_threshold}_dm{100*distance_multiplier}.hdf5"
elif a == '7':
  print("FV = [ supervised , moco imagenet , moco places2 ]")
  output_path =   output_path = f"/scratch/EVM_{distance_function}_model_imagenet_b3_joint3_tail{tailsize}_ct{100*cover_threshold}_dm{100*distance_multiplier}.hdf5"
else:
  raise ValueError()
  output_path = None

print("\nStart\n\n")

device='cuda'

startTime = time.time()

#Load features and labels
known = np.load(data_path,allow_pickle=True)
known = known[~np.isnan(known).any(axis=1)]

L = known[:,0]
if a == '1':
  FV = known[:,1:1537]
if a == '2':
  FV = known[:,1537:3073]
if a == '3':
  FV = known[:,3073:4609]
if a == '4':
  FV = known[:,1537:4609]
if a == '5':
  FV = known[:,1:3073]
if a == '6':
  FV = np.hstack((known[:,1:1537],known[:,3073:4609]))
if a == '7':
  FV = known[:,1:]
del known


labels = []
features = []
for i in range(len(FV)):
  labels.append(int(L[i]))
  features.append(FV[i])

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
