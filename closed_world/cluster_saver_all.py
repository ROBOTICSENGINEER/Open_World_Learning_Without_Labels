import time
import numpy as np
import pandas as pd
import os
import PIL
from collections import OrderedDict
import cv2
import json
import argparse
import pickle
import copy
import gc
from finch import FINCH
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from timm.models.efficientnet import efficientnet_b3 as net_model_from_lib  # timm library

import sys


t0 = time.time()


batch_size = 256

config = json.load(open("config_owl_b3_sp.json"))

number_of_known_classes = config["number_of_known_classes"]
cnn_path_supervised = config["cnn_path_supervised"]
cnn_path_moco_places = config["cnn_path_moco_places"]

image_size = config["image_size"]
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])

N_feature = 1536 * 2
n_cpu = 32


assert os.path.isfile(cnn_path_supervised) 
assert os.path.isfile(cnn_path_moco_places) 


class csv_data_class(Dataset):

  def __init__(self, path, transform_supervised, transform_moco):
    with open(path) as f:
      self.samples = [line.rstrip() for line in f if line != '']
    self.transform_supervised = transform_supervised
    self.transform_moco = transform_moco

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    A, L = S.split(',')
    img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(img)
    x_supervised = self.transform_supervised(img_pil)
    x_moco = self.transform_moco(img_pil)
    y = int(L)
    return (x_supervised, x_moco, y, A)




image_transform_supervised = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])

image_transform_moco = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])


t1 = time.time()

cnn_model_supervised = net_model_from_lib(num_classes = number_of_known_classes)  # timm library
cnn_model_moco_places = net_model_from_lib(num_classes = number_of_known_classes)  # timm library


assert os.path.isfile(cnn_path_supervised)
checkpoint = torch.load(cnn_path_supervised)
if 'epoch' in checkpoint.keys():
  state_dict_model = checkpoint['state_dict']
else:
  state_dict_model = checkpoint
from collections import OrderedDict
new_state_dict_model = OrderedDict()
for k, v in state_dict_model.items():
  if 'module.' == k[:7]: 
    name = k[7:] # remove `module.`
    new_state_dict_model[name] = v
  else:
     new_state_dict_model[k] = v
cnn_model_supervised.load_state_dict(new_state_dict_model)
for parameter in cnn_model_supervised.parameters():
  parameter.requires_grad = False
cnn_model_supervised.cuda()
cnn_model_supervised.eval()


assert os.path.isfile(cnn_path_moco_places)
checkpoint = torch.load(cnn_path_moco_places, map_location="cpu")
# rename moco pre-trained keys
print('keys = ', checkpoint.keys())
state_dict = checkpoint['state_dict']
for k in list(state_dict.keys()):
  # retain only encoder_q up to before the embedding layer
  if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    # remove prefix
    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
  # delete renamed or unused k
  del state_dict[k]
msg = cnn_model_moco_places.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
for parameter in cnn_model_moco_places.parameters():
  parameter.requires_grad = False
cnn_model_moco_places.cuda()
cnn_model_moco_places.eval()



device = torch.device('cuda')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  cnn_model_supervised = torch.nn.DataParallel(cnn_model_supervised)
  cnn_model_moco_places = torch.nn.DataParallel(cnn_model_moco_places)
cnn_model_supervised.eval()
cnn_model_moco_places.eval()

t2 = time.time()
print("loading CNN time = ", t2 - t1)


torch.backends.cudnn.benchmark=True



csv_name_dict = { 0:'imagenet_test_v2_73.csv',  1:'imagenet_test_v2_85.csv',
                  2:'imagenet_test_v2_93.csv' , 3:'imagenet_1000_val.csv', 
                  4:'imagenet_166.csv'}
              
  
name_dict = { 0:'test_hard', 1:'test_medium', 2:'test_easy', 3:'val', 4:'unknown'}
  
with torch.no_grad():
  for k in range(5):
    
    t4 = time.time()
    
    
    csv_name = csv_name_dict[k]
    data_name = name_dict[k]
    
    print(f"start extracting address in {data_name} datasets")
    
    data_set = csv_data_class(path = f'/scratch/mjafarzadeh/' + csv_name,
                              transform_supervised = image_transform_supervised, 
                              transform_moco = image_transform_moco)
    data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
    N_data = data_set.__len__()
    feature_storage = torch.empty([N_data, N_feature+1], requires_grad = False)
    n = 0
    image_list = []
    for i, (x_supervised, x_moco, y, A) in enumerate(data_loader, 0):
      x_supervised = x_supervised.cuda()
      x_moco = x_moco.cuda()
      FV_supervised, _ = cnn_model_supervised(x_supervised)
      FV_moco_places, _ = cnn_model_moco_places(x_moco)
      FV = torch.cat((FV_supervised, FV_moco_places), 1)
      feature_storage[n:(n+FV.size(0)), 0] = y
      feature_storage[n:(n+FV.size(0)), 1:] = FV.cpu()
      image_list = image_list + list(A)
      n = n + FV.size(0)
    
    assert len(image_list) == n
    
    
    data_ = feature_storage.detach().clone().cpu().numpy()
    data_ = data_[:n,:]
    
    feature =  data_[:,1:]
    true_label =  data_[:,0]
    c_all, num_clust, req_c = FINCH(feature)
    
    cluster_labels = c_all[:,1]
    number_of_clusters = num_clust[1]  # number of clusters after clustering. 
    
    storage_dict = dict()
    for L, K, A in zip(true_label, cluster_labels, image_list):
      if K in storage_dict:
        storage_dict[K].append( (L, A) )
      else:
        storage_dict[K] = [ (L, A) ]
    

    torch.save(storage_dict, f'/scratch/mjafarzadeh/clusters_all_b3_SP_{data_name}.pth')
    del feature_storage, data_, storage_dict
    t5 = time.time()
    print(f"extracting {data_name} time = ", t5 - t4)

t6 = time.time()
print("total time = ", t6-t0)
print('\nEnd')
