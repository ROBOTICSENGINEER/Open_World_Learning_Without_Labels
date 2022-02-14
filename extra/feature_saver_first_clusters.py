import time
import numpy as np
import os
import PIL
from collections import OrderedDict
import cv2
import json
import copy
import gc
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from timm.models.efficientnet import efficientnet_b3 as net_model_from_lib  # timm library




t0 = time.time()

output_folder = f"/scratch/mjafarzadeh/first_features/"

batch_size = 256

number_of_known_classes = 1000
cnn_path_supervised = "/scratch/mjafarzadeh/trained_efficientnet_b3_fp16_imagenet.pth"
cnn_path_moco_places = "/scratch/mjafarzadeh/moco_places2_0199.pth"


n_cpu = 32

image_size = 300
N_feature = 1536 * 2
np.random.seed(2)
torch.manual_seed(2)


assert os.path.isfile(cnn_path_supervised) 
assert os.path.isfile(cnn_path_moco_places) 


class list_data_class(Dataset):

  def __init__(self, image_list, transform_supervised, transform_moco):
    self.samples = image_list
    self.transform_supervised = transform_supervised
    self.transform_moco = transform_moco

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    A = self.samples[index]
    img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(img)
    x_supervised = self.transform_supervised(img_pil)
    x_moco = self.transform_moco(img_pil)
    return (x_supervised, x_moco)





image_transform_supervised = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])

image_transform_moco = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])


t1 = time.time()

model_supervised = net_model_from_lib(num_classes = number_of_known_classes)  # timm library
model_moco = net_model_from_lib(num_classes = number_of_known_classes)  # timm library


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
model_supervised.load_state_dict(new_state_dict_model)
for parameter in model_supervised.parameters():
  parameter.requires_grad = False
model_supervised.cuda()
model_supervised.eval()


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
msg = model_moco.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
for parameter in model_moco.parameters():
  parameter.requires_grad = False
model_moco.cuda()
model_moco.eval()


t2 = time.time()
print("loading CNN time = ", t2 - t1)

torch.backends.cudnn.benchmark=True

with torch.no_grad():
  for clustering_type in ["finch", "facto"]:
    for level in ["easy" , "hard"]:
      for id in range(1,6):
        for with_or_without in ["with" , "without"]:
          print(f"{clustering_type} , test {level} {id}, {with_or_without} knowns")
          
          folder = f"/scratch/mjafarzadeh/result_first_{clustering_type}/"
          file = folder + f"{clustering_type}_{with_or_without}_known_{level}_{id}_first_clusters.txt"
          output_name = f"feature_first_clusters_{clustering_type}_{with_or_without}_known_{level}_{id}.pth"
          
          storage_dict = dict()
      
          cluster_image_address_dict = dict()
          N_cluster = 0
          
          with open(file, 'r') as f:
            for L in f:
              line = L.strip('\n')
              if len(line)>4:
                if 'cluster' in line:
                  cluster = int(line.split()[1])
                  cluster_image_address_dict[cluster] = list()
                  N_cluster = N_cluster + 1
                elif 'ImageNet' in line:
                  cluster_image_address_dict[cluster].append('/scratch/datasets/'  + line)
                else:
                  raise ValueError()
                    
              
          for cluster in range(1 , 1 + N_cluster):
            #print(f"{cluster = }")
            current_dict = dict()
            images_list = cluster_image_address_dict[cluster]
            
            
            
            images_list_all = [A[18:] for A in images_list]
            images_list_known = [A for A in images_list_all if 'ImageNet/v2_' in A]
            images_list_unknown = [A for A in images_list_all if not 'ImageNet/v2_' in A]
            assert len(images_list) == len(images_list_known) + len(images_list_unknown)
            
            current_dict['address'] = images_list_all
      
            
            uni_known = set()
            uni_unknown = set()
            for A in images_list_known:
              wnid = A.split('/')[-2]
              uni_known.add(wnid)
            for A in images_list_unknown:
              wnid = A.split('/')[-2]
              uni_unknown.add(wnid)
            
            
            uni_known = list(uni_known)
            uni_unknown = list(uni_unknown)
            uni_all = uni_known + uni_unknown
            
            current_dict['wnid_all'] = uni_all
            current_dict['wnid_known'] = uni_known
            current_dict['wnid_unknown'] = uni_unknown
            
            
            
            current_dict['N_images'] = len(images_list)
            current_dict['N_known_images'] = len(images_list_known)
            current_dict['N_unknown_images'] = len(images_list_unknown)
            current_dict['R_known_images'] = len(images_list_known) / len(images_list)
            current_dict['R_unknown_images'] = len(images_list_unknown)/ len(images_list)
            current_dict['N_classes'] = len(uni_all)
            current_dict['N_known_classes'] = len(uni_known)
            current_dict['N_unknown_classes'] = len(uni_unknown)
            
            data_set = list_data_class(image_list = images_list,
                                transform_supervised = image_transform_supervised, 
                                transform_moco = image_transform_moco)
            data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
            N_data = data_set.__len__()
          
            for batch_id, (x_supervised, x_moco) in enumerate(data_loader, 0):
              assert batch_id == 0
              x_supervised = x_supervised.cuda()
              x_moco = x_moco.cuda()
              F_supervised, _ = model_supervised(x_supervised)
              F_moco, _ = model_moco(x_moco)
              FV = torch.cat((F_supervised, F_moco), 1)
              FV = FV.cpu()
            current_dict['feature'] = FV.detach().clone()
            storage_dict[cluster] = copy.deepcopy(current_dict)
          #print(storage_dict)
          torch.save(storage_dict, output_folder + output_name)
