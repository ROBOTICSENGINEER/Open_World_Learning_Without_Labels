import time
import numpy as np
import pandas as pd
import os
import PIL
from collections import OrderedDict
from finch import FINCH
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


parser = argparse.ArgumentParser(description='Open World Learning')
parser.add_argument('--config', required=True, help='Path to config json file.')
parser.add_argument('--test_csv', required=True, help='csv file of a test.')
parser.add_argument('--result_csv', required=True, help='csv file name to save the result.')
parser.add_argument('--no-adapt', action='store_true', help='no adaptation.')
                    

args_owl = parser.parse_args()

assert os.path.isfile(args_owl.config) 

config = json.load(open(args_owl.config))

batch_size = 100 


test_csv_path = config["test_csv_dir"] + args_owl.test_csv
result_path = config["result_dir"] + args_owl.result_csv
number_of_known_classes = config["number_of_known_classes"]
cnn_path_supervised = config["cnn_path_supervised"]
cnn_path_moco_imagenet = config["cnn_path_moco_imagenet"]
evm_path = config["evm_path"]
feature_known_path = config["feature_known_path"]
cover_threshold = config["cover_threshold"]
distance_multiplier = config["distance_multiplier"]
unknown_distance_multiplier = config["unknown_distance_multiplier"]
tail_size = config["tail_size"]
distance_metric = config["distance_metric"]
chunk_size = config["chunk_size"]
n_cpu = config["n_cpu"]
image_size = config["image_size"]
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])
test_size = config["test_size"]
min_number_point_to_start_clustering = config["min_number_point_to_start_clustering"]
min_number_cluster_to_start_adaptation = config["min_number_cluster_to_start_adaptation"]
min_number_point_to_create_evm = config["min_number_point_to_create_evm"]

assert os.path.isfile(test_csv_path) 
assert os.path.isfile(cnn_path_supervised) 
assert os.path.isfile(cnn_path_moco_imagenet) 
assert os.path.isfile(evm_path) 
assert os.path.isdir(config["result_dir"]) 
assert test_size % batch_size == 0

image_transform_supervised = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["transformer_mean_supervised"], std=config["transformer_std_supervised"]) ])

image_transform_moco = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=config["transformer_mean_moco"], std=config["transformer_std_moco"]) ])


class labeling(object):
  def __init__(self, number_of_known = 1000):
    self.current_label = dict()
    self.folder_to_number_dict = json.load(open("./data/folder_to_id_dict_all.json"))
    self.last_index = number_of_known
    self.number_of_known = number_of_known

  def get_label(self, x):
    z_train = max(x.find('v2_73/') ,  x.find('train/'))
    z_val = x.find('val_in_folders/')
    if z_train>=0:
      ind = int(   self.folder_to_number_dict [     x[(z_train+6):(z_train+15)]   ]  )
    elif z_val>=0:
      ind = int(   self.folder_to_number_dict [     x[(z_val+15):(z_val+24)]   ]  )
    else:
      raise ValueError(f"x is not in folder_to_number_dict")
    assert ind > 0
    if ind <= self.number_of_known:
      return ind
    # else: it is unknown
    if ind in self.current_label:
      return self.current_label[ind]
    else:
      self.last_index = self.last_index + 1
      self.current_label[ind] = self.last_index
      return self.last_index
      
feedback = labeling()


class csv_data_class(Dataset):

  def __init__(self, path, transform_supervised, transform_moco):
    with open(path) as f:
      self.samples = [line.rstrip() for line in f if line != '']
      assert len(self.samples) == test_size
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
    i = A.find('ImageNet/')
    address = A[i:]
    #y = int(L)
    #return (x,y)
    return (address, x_supervised, x_moco)


dataset_test = csv_data_class(path = test_csv_path,
                              transform_supervised = image_transform_supervised, 
                              transform_moco = image_transform_moco)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_cpu)



t1 = time.time()

cnn_model_supervised = net_model_from_lib(num_classes = number_of_known_classes)  # timm library
cnn_model_moco_imagenet = net_model_from_lib(num_classes = number_of_known_classes)  # timm library


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


assert os.path.isfile(cnn_path_moco_imagenet)
checkpoint = torch.load(cnn_path_moco_imagenet, map_location="cpu")
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
msg = cnn_model_moco_imagenet.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
for parameter in cnn_model_moco_imagenet.parameters():
  parameter.requires_grad = False
cnn_model_moco_imagenet.cuda()
cnn_model_moco_imagenet.eval()


t2 = time.time()
print("Loading cnn time = ", t2 - t1)


args_evm  = argparse.Namespace()
args_evm.cover_threshold = [cover_threshold]
args_evm.distance_multiplier = [distance_multiplier]
args_evm.tailsize = [tail_size]
args_evm.distance_metric = distance_metric
args_evm.chunk_size = chunk_size

args_evm_unknown = copy.deepcopy(args_evm)
args_evm_unknown.distance_multiplier = [unknown_distance_multiplier]

with torch.no_grad():
  evm_model = pickle.load( open(evm_path , "rb" ) )
  
  t3 = time.time()
  print("Loading evm time = ", t3 - t2)
  
  if not args_owl.no_adapt: 
    data_train_evm =  torch.from_numpy(  np.load(feature_known_path) )
    features_dict_train = OrderedDict()
    for k in range(1, number_of_known_classes+1):
      F = data_train_evm[data_train_evm[:,0]==k]
      features_dict_train[k] = F[:,1:].detach().clone()#.cuda()
    t31 = time.time()
    print("Loading feature known time = ", t31 - t3)
  
  torch.backends.cudnn.benchmark=True

  t4 = time.time()
  number_of_discovered_classes = 0 
  residual_dict = dict()
  clustered_dict = dict()
  probability_list = [0] * int(test_size/batch_size)
  image_list = [] 
  for i, (image_names, x_supervised, x_moco) in enumerate(test_loader, 0):
    t5 = time.time()
    print("\nbatach ", i+1)
    x_supervised = x_supervised.cuda()
    x_moco = x_moco.cuda()
    FV_supervised, Logit = cnn_model_supervised(x_supervised)
    FV_moco_imagenet, Logit = cnn_model_moco_imagenet(x_moco)
    del x_supervised, x_moco, Logit
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    FV_supervised = FV_supervised.cpu()
    FV_moco_imagenet = FV_moco_imagenet.cpu()
    FV = torch.cat((FV_supervised, FV_moco_imagenet), 1)
    feature_dict = OrderedDict()
    feature_dict[0] = FV.double()
    Pr_iterator = EVM_Inference([0], feature_dict, args_evm, 0, evm_model)
    for j,pr in enumerate(Pr_iterator):
      prob  = pr[1][1]#.cuda()
      assert j == 0
    del Pr_iterator, pr
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    n = 1 + number_of_known_classes + number_of_discovered_classes
    probability_tensor = torch.zeros(prob.shape[0], n)
    probability_tensor[:,1:] = prob
    P_max_all , _  = torch.max(prob, axis=1)
    pu = 1 -  P_max_all
    probability_tensor[:,0] = pu
    norm = torch.norm(probability_tensor, p=1, dim=1)
    normalized_tensor = probability_tensor/norm[:,None]
    probability_list[i] = normalized_tensor.clone().cpu()
    image_list = image_list + list(image_names)
    t6 = time.time()
    print("classification time = ", t6 - t5)
    
    
    if not args_owl.no_adapt:
      nu = 0
      class_to_process = set()
      
      for j, im_name in enumerate(image_names):
        label = feedback.get_label(im_name)
        if label > number_of_known_classes:
          class_to_process.add(label)
          feature_j = FV[j:(j+1),:].clone().double()
          if label in features_dict_train:
            features_dict_train[label] = torch.cat((features_dict_train[label], feature_j), 0)
          else:
            nu = nu + 1
            features_dict_train[label] = feature_j
      
      
      if len(class_to_process) > 0:
        class_to_process = sorted(list(class_to_process))
        list_of_tuples = [0.0] * len(class_to_process)
        evm_iterator_i = EVM_Training(class_to_process, features_dict_train, args_evm_unknown, 0)
        evm_counter = 0
        for evm in enumerate(evm_iterator_i):
          # label = evm[1][1][0]
          # mini_evm = evm[1][1][1]
          list_of_tuples[evm_counter] = (evm[1][1][0], evm[1][1][1])
          evm_counter = evm_counter + 1

        for j,class_number in enumerate(class_to_process):
          # evm is an OrderedDict
          assert list_of_tuples[j][0] == class_number
          evm_model[class_number] = list_of_tuples[j][1]
      
        del evm_iterator_i, evm
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

      number_of_discovered_classes = number_of_discovered_classes + nu
      print(f"{nu} new evm classes added.")
      print(f"len(clustered_dict) = {len(clustered_dict)}")
      print(f"len(residual_dict) = {len(residual_dict)}")
      print(f"Total discovered classes = {number_of_discovered_classes}")
      gc.collect()
      torch.cuda.empty_cache()
      torch.cuda.ipc_collect()
      t7 = time.time()
      print("adaptation time = ", t7 - t6)
      print("batch time = ", t7 - t5)

  t8 = time.time()
  print("loop time = ", t8 - t4)

  normalized_tensor = torch.zeros(test_size,probability_list[-1].shape[1])

  n1 = 0
  for k,p in enumerate(probability_list):
    j = p.shape[1]
    n2 = n1 + p.shape[0]
    normalized_tensor[n1:n2,:j] = p
    n1 = n2

  col = ["name"] + [f"c{k}" for k in range(probability_list[-1].shape[1])]
  df_characterization = pd.DataFrame(zip(image_list,*normalized_tensor.t().tolist()), columns=col)
  df_characterization.to_csv(result_path, index = False, header = False, float_format='%.4f')
  t9 = time.time()
  print("saving time = ", t9 - t8)

t10 = time.time()
print("\ntotal time = ", t10 - t0)
print("End\n")

