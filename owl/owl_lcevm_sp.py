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
parser.add_argument('--output_name', required=True, help='output name to save the result.')
parser.add_argument('--no-adapt', action='store_true', help='no adaptation.')


args_owl = parser.parse_args()

assert os.path.isfile(args_owl.config) 

config = json.load(open(args_owl.config))

batch_size = 100 

svm_path = config["svm_path"]
test_csv_path = config["test_csv_dir"] + args_owl.test_csv
number_of_known_classes = config["number_of_known_classes"]
cnn_path_supervised = config["cnn_path_supervised"]
cnn_path_moco_places = config["cnn_path_moco_places"]
linear_path = config["linear_path"]
evm_path = config["evm_path"]
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
threshold_softmax = config["threshold_sm"]
feature_size = config["feature_size"]
min_number_point_to_start_clustering = config["min_number_point_to_start_clustering"]
min_number_cluster_to_start_adaptation = config["min_number_cluster_to_start_adaptation"]
min_number_point_to_create_class = config["min_number_point_to_create_class"]


assert os.path.isfile(svm_path)
assert os.path.isfile(test_csv_path) 
assert os.path.isfile(cnn_path_supervised) 
assert os.path.isfile(cnn_path_moco_places) 
assert os.path.isfile(evm_path) 
assert os.path.isdir(config["result_dir"]) 
assert test_size % batch_size == 0


class check_cluster_quality(object):
  
  def __init__(self, path_svm):
    self.svm = pickle.load( open( path_svm, "rb" ) )
  
  def run(self, feature):
    N = feature.shape[0]
    center = torch.mean(feature, dim = 0)
    center = center.repeat(N,1)
    de = feature - center
    VE = torch.linalg.matrix_norm(de, ord = 'fro')**2 / (N-1)

    norm_2 = torch.norm(feature, p=2, dim=1)
    FV = feature/norm_2[:,None]
    center = torch.mean(FV, dim = 0)
    norm_2 = torch.linalg.vector_norm(center, ord=2)
    center = center / norm_2
    d = 1 - torch.mm(FV,center.view(-1,1))
    VC = torch.linalg.matrix_norm(d, ord = 'fro')**2 / (N-1)

    x = np.array([[VE.item(),VC.item()]])
    y =  self.svm.predict(x)
    
    return y
    
    
cluster_quality_checker = check_cluster_quality(svm_path)


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
    return (address, x_supervised, x_moco)


dataset_test = csv_data_class(path = test_csv_path,
                              transform_supervised = image_transform_supervised, 
                              transform_moco = image_transform_moco)
test_loader = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, num_workers=n_cpu)


SoftMax = torch.nn.Softmax(dim=1).cuda()

class Linear_Classifier(torch.nn.Module):
  def __init__(self, num_classes):
    super(Linear_Classifier, self).__init__()
    self.fc = torch.nn.Linear(feature_size, num_classes)

  def forward(self, x):
    return self.fc(x)

t1 = time.time()

cnn_model_supervised = net_model_from_lib(num_classes = number_of_known_classes)  # timm library
cnn_model_moco_places = net_model_from_lib(num_classes = number_of_known_classes)  # timm library
linear_model = Linear_Classifier(num_classes = number_of_known_classes) 

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
#print('keys = ', checkpoint.keys())
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


assert os.path.isfile(linear_path)
checkpoint = torch.load(linear_path)
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
linear_model.load_state_dict(new_state_dict_model)
for parameter in linear_model.parameters():
  parameter.requires_grad = False
linear_model.cuda()
linear_model.eval()


t2 = time.time()
print("Loading cnn time = ", t2 - t1)


args_evm  = argparse.Namespace()
args_evm.cover_threshold = [cover_threshold]
args_evm.distance_multiplier = [distance_multiplier]
args_evm.tailsize = [tail_size]
args_evm.distance_metric = distance_metric
args_evm.chunk_size = chunk_size

args_evm_incremental = copy.deepcopy(args_evm)
args_evm_incremental.distance_multiplier = [unknown_distance_multiplier]


torch.backends.cudnn.benchmark=True
with torch.no_grad():
  evm_model = pickle.load( open(evm_path , "rb" ) )
  t4 = time.time()
  print("Loading evm time = ", t4 - t2)
  number_of_discovered_classes = 0 
  residual_dict = dict()
  clustered_dict = dict()
  features_dict_incremental = OrderedDict()
  evm_model_incremental = OrderedDict()
  probability_list = [0] * int(test_size/batch_size)
  raw_list = [0] * int(test_size/batch_size)
  image_list = [] 
  for i, (image_names, x_supervised, x_moco) in enumerate(test_loader, 0):
    t5 = time.time()
    print("\nbatch ", i+1)
    x_supervised = x_supervised.cuda()
    x_moco = x_moco.cuda()
    FV_supervised, Logit = cnn_model_supervised(x_supervised)
    FV_moco_places, Logit = cnn_model_moco_places(x_moco)
    del x_supervised, x_moco, Logit
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.ipc_collect()
    FV = torch.cat((FV_supervised,  FV_moco_places), 1)

    Logit = linear_model(FV)
    softmax = SoftMax(Logit.double())
    FV = FV.double()
    
    sm , _  = torch.max(softmax, axis=1)
    predicted_known = (sm >= threshold_softmax)
    predicted_unknown = ~predicted_known
    
    n = 1 + number_of_known_classes + number_of_discovered_classes
    probability_tensor = torch.zeros(batch_size, n, dtype= torch.double)
    probability_tensor[predicted_known,1:(1+number_of_known_classes)] = softmax[predicted_known,:].cpu()
    
    
    if torch.sum(predicted_unknown) > 0:
      if number_of_discovered_classes > 0:
        FV_predicted_unknown = FV [predicted_unknown,:]
        feature_dict_predicted_unknown = OrderedDict()
        feature_dict_predicted_unknown[0] = FV_predicted_unknown.double()
        Pr_iterator_predicted_unknown = EVM_Inference([0], feature_dict_predicted_unknown, args_evm_incremental, 0, evm_model_incremental)
        for j_predicted_unknown,pr_predicted_unknown in enumerate(Pr_iterator_predicted_unknown):
          prob_predicted_unknown  = pr_predicted_unknown[1][1]#.cuda()
          assert j_predicted_unknown == 0
        del Pr_iterator_predicted_unknown, pr_predicted_unknown
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        P_max_predicted_unknown , _  = torch.max(prob_predicted_unknown, axis=1)
        pu_predicted_unknown = 1 -  P_max_predicted_unknown
        probability_tensor[predicted_unknown,0] = pu_predicted_unknown
        probability_tensor[predicted_unknown,1 + number_of_known_classes : ] = prob_predicted_unknown
      else:
        probability_tensor[predicted_unknown,0] = 1.0
    
    norm = torch.norm(probability_tensor, p=1, dim=1)
    normalized_tensor = probability_tensor/norm[:,None]
    probability_list[i] = normalized_tensor.detach().clone().cpu()
    image_list = image_list + list(image_names)
    t6 = time.time()
    print("classification time = ", t6 - t5)
    
    FV = FV.cpu()
    if not args_owl.no_adapt:
      nu = 0 
      p_max , i_max  = torch.max(normalized_tensor, axis=1)
      for k in range(normalized_tensor.shape[0]):
        if i_max[k] == 0 :  # predicted unnkwon unknown
          residual_dict[image_names[k]] = FV[k,:].numpy()
      print(f"len(residual_dict) = {len(residual_dict)}")
      
      if len(residual_dict) >= min_number_point_to_start_clustering:
        image_names_residual, FVs_residual = zip(*residual_dict.items())
        data =  np.vstack(list(FVs_residual))
        c_all, num_clust, req_c = FINCH(data)
        if len(num_clust) <= 2:
          index_partition_selected = 0
        else:
          index_partition_selected = 1
        
        cluster_labels = c_all[:,index_partition_selected]
        number_of_clusters = num_clust[index_partition_selected]  # number of clusters after clustering. 
        
        to_be_delete = []
        if number_of_clusters >= min_number_cluster_to_start_adaptation:
          class_to_process = []
          for cluster_number in range(number_of_clusters):  # number of clusters after clustering.
            index = [iii for iii in range(len(cluster_labels)) if cluster_labels[iii] == cluster_number]
            if len(index) >= min_number_point_to_create_class:
              feature_this_cluster = torch.from_numpy(np.array([FVs_residual[jjj] for jjj in index]))
              if cluster_quality_checker.run(feature_this_cluster) == 1:
              
                to_be_delete = to_be_delete + index
                nu = nu + 1
                class_number = int(nu+number_of_discovered_classes+number_of_known_classes)
                features_dict_incremental[class_number] = (torch.from_numpy(np.array([FVs_residual[jjj] for jjj in index]))).double()#.cuda()
                class_to_process.append(class_number)
        
          if (  ((len(class_to_process) > 0) and (number_of_discovered_classes > 0))
                or (len(class_to_process) > 2) ):
        
            list_of_tuples = [0.0] * len(class_to_process)
            evm_iterator_incremental = EVM_Training(class_to_process, features_dict_incremental, args_evm_incremental, 0)
            evm_counter = 0
            for evm in enumerate(evm_iterator_incremental):
              # label = evm[1][1][0]
              # mini_evm = evm[1][1][1]
              list_of_tuples[evm_counter] = (evm[1][1][0], evm[1][1][1])
              evm_counter = evm_counter + 1
      
            for j,class_number in enumerate(class_to_process):
              # evm is an OrderedDict
              assert list_of_tuples[j][0] == class_number
              evm_model_incremental[class_number] = list_of_tuples[j][1]
            
            
            del evm_iterator_incremental, evm
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            
          else:
            nu = 0 
            to_be_delete = []
        

          if nu > 0:
            image_covered = []
            for j in (to_be_delete):
              image_covered.append(image_names_residual[j])
            for name in image_covered:
              fv_name = residual_dict[name]
              clustered_dict.update({name:fv_name})
              del residual_dict[name]
          number_of_discovered_classes = number_of_discovered_classes + nu
      print(f"{nu} new classes added.")
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

  result_path = config["result_dir"] + args_owl.output_name + f'_output.csv'
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

