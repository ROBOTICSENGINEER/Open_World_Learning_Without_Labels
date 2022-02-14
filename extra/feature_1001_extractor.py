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
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from timm.models.efficientnet import efficientnet_b3 as net_model_from_lib  # timm library
from EVM import EVM_Training , EVM_Inference


import weibull
import sys
sys.modules['weibull'] = weibull

t0 = time.time()


data_list = [ ('train', 'imagenet_1000_train'), 
              ('val', 'imagenet_1000_val'), 
              ('test_hard', 'imagenet_test_v2_73' ), 
              ('test_medium' , 'imagenet_test_v2_85'), 
              ('test_easy' , 'imagenet_test_v2_93')]


config = json.load(open('config_owl_b3_supervised_mocoPlaces.json'))

batch_size = 256


number_of_known_classes = config["number_of_known_classes"]
cnn_path_supervised = config["cnn_path_supervised"]
cnn_path_moco_places = config["cnn_path_moco_places"]
evm_path = config["evm_path"]
feature_known_path = config["feature_known_path"]
cover_threshold = config["cover_threshold"]
distance_multiplier = config["distance_multiplier"]
tail_size = config["tail_size"]
distance_metric = config["distance_metric"]
chunk_size = config["chunk_size"]
n_cpu = config["n_cpu"]
image_size = config["image_size"]
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])


assert os.path.isfile(cnn_path_supervised) 
assert os.path.isfile(cnn_path_moco_places) 
assert os.path.isfile(evm_path) 


image_transform_supervised = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["transformer_mean_supervised"], std=config["transformer_std_supervised"]) ])

image_transform_moco = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["transformer_mean_moco"], std=config["transformer_std_moco"]) ])



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
    return (x_supervised, x_moco, y)


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



if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  cnn_model_supervised = torch.nn.DataParallel(cnn_model_supervised)
  cnn_model_moco_places = torch.nn.DataParallel(cnn_model_moco_places)

t2 = time.time()
print("Loading cnn time = ", t2 - t1)

args_evm  = argparse.Namespace()
args_evm.cover_threshold = [cover_threshold]
args_evm.distance_multiplier = [distance_multiplier]
args_evm.tailsize = [tail_size]
args_evm.distance_metric = distance_metric
args_evm.chunk_size = chunk_size


torch.backends.cudnn.benchmark=True
with torch.no_grad():
  evm_model = pickle.load( open(evm_path , "rb" ) )
  
  t3 = time.time()
  print("Loading evm time = ", t3 - t2)
  
  for data_name, csv_name in data_list:
    
    csv_path = "/scratch/mjafarzadeh/" + csv_name + ".csv"
    
    t4 = time.time()
    print("start extracting ", data_name )
  
    dataset_csv = csv_data_class(path = csv_path,
                                  transform_supervised = image_transform_supervised, 
                                  transform_moco = image_transform_moco)                              
                                
    data_loader = DataLoader(dataset=dataset_csv, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
  
    N_data = dataset_csv.__len__()
  
    feature_tensor = torch.empty([N_data, 3072+1])
    n = 0
    for i, (x_supervised, x_moco, y) in enumerate(data_loader, 0):
      x_supervised = x_supervised.cuda()
      x_moco = x_moco.cuda()
      FV_supervised, _ = cnn_model_supervised(x_supervised)
      FV_moco_places, _ = cnn_model_moco_places(x_moco)
      FV = torch.cat((FV_supervised, FV_moco_places), 1)
      feature_dict = OrderedDict()
      feature_dict[0] = FV.double()
      Pr_iterator = EVM_Inference([0], feature_dict, args_evm, 0, evm_model)
      for j,pr in enumerate(Pr_iterator):
        prob  = pr[1][1]#.cuda()
        assert j == 0
      del Pr_iterator, pr
      _ , i_knowns  = torch.max(prob, axis=1)
      i_knowns = i_knowns + 1
      probability_tensor = torch.zeros(prob.shape[0], 1001)
      probability_tensor[:,1:] = prob
      P_max_all , _  = torch.max(prob, axis=1)
      pu = 1 -  P_max_all
      probability_tensor[:,0] = pu
      norm = torch.norm(probability_tensor, p=1, dim=1)
      normalized_tensor = probability_tensor/norm[:,None]
      p_max , i_max  = torch.max(normalized_tensor, axis=1)
      index = (i_max==0) * (i_knowns != y)
      y[index] = 1001
      feature_tensor[n:(n+x_supervised.size(0)), 0] = y
      feature_tensor[n:(n+x_supervised.size(0)), 1:] = FV.cpu()
      n = n+x_supervised.size(0)
    
    is_bad =  (feature_tensor[:,0] == 1001)
    feature_tensor_bad = feature_tensor[is_bad]
    feature_tensor_bad = feature_tensor_bad[~torch.any(feature_tensor_bad.isnan(),dim=1)]
    feature_tensor_rest = feature_tensor[~is_bad]
    feature_tensor_rest = feature_tensor_rest[~torch.any(feature_tensor_rest.isnan(),dim=1)]
    
    torch.save(feature_tensor_bad, f'/scratch/mjafarzadeh/feature_bad_b3_SP_{data_name}.pth')
    torch.save(feature_tensor_rest, f'/scratch/mjafarzadeh/feature_rest_b3_SP_{data_name}.pth')
    
    del feature_tensor, feature_tensor_rest, feature_tensor_bad
    
    t5 = time.time()
    print(f"extracting {data_name} data set time = ", t5 - t4)



t8 = time.time()
print("total time = ", t8 - t0)

print('\nEnd')
