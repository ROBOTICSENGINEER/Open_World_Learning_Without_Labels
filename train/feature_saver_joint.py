import numpy as np
import h5py
import cv2
import PIL 
from random import shuffle
import torch
from torchsummary import summary
from torch import from_numpy, tensor
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import time
from ..timm.models.efficientnet import efficientnet_b3  # timm library

model_name = 'joint_supervised_moco_b3'
address_of_trained_model_0 = '/scratch/trained_efficientnet_b3_fp16_imagenet.pth.tar'
address_of_trained_model_1 = '/scratch/moco_imagenet_0199.pth.tar'
address_of_trained_model_2 = '/scratch/moco_places2_0199.pth.tar'


n_cpu = 32
batch_size = 512
image_size = 300
N_feature = 1536 * 3
Number_of_Classes = 1000

np.random.seed(2)
torch.manual_seed(2)



class known_train_data_class(Dataset):

  def __init__(self, transform_supervised, transform_moco):
    with open('../data/imagenet_1000_train.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform_supervised = transform_supervised
    self.transform_moco = transform_moco


  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x_supervised = self.transform_supervised(img_pil)
      x_moco = self.transform_moco(img_pil)
      y = int(L)-1
    return (x_supervised,x_moco,y)


class known_val_data_class(Dataset):

  def __init__(self, transform_supervised, transform_moco):
    with open('../data/imagenet_1000_val.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform_supervised = transform_supervised
    self.transform_moco = transform_moco

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x_supervised = self.transform_supervised(img_pil)
      x_moco = self.transform_moco(img_pil)
      y = int(L)-1
    return (x_supervised,x_moco,y)



class unknown_train_data_class(Dataset):

  def __init__(self, transform_supervised, transform_moco):
    with open('../data/imagenet_166.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform_supervised = transform_supervised
    self.transform_moco = transform_moco

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x_supervised = self.transform_supervised(img_pil)
      x_moco = self.transform_moco(img_pil)
      y = int(L)-1
    return (x_supervised,x_moco,y)



image_transform_val_supervised = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])
            
image_transform_val_moco = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

XY_train = known_train_data_class(transform_supervised  = image_transform_val_supervised , transform_moco  = image_transform_val_moco)
XY_val = known_val_data_class(transform_supervised  = image_transform_val_supervised , transform_moco  = image_transform_val_moco)
XY_unknown = unknown_train_data_class(transform_supervised  = image_transform_val_supervised , transform_moco  = image_transform_val_moco)

train_loader = DataLoader(dataset=XY_train, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
val_loader = DataLoader(dataset=XY_val, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
unknown_loader = DataLoader(dataset=XY_unknown, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

model_0 = efficientnet_b3(num_classes=Number_of_Classes)  #timm library
model_1 = efficientnet_b3(num_classes=Number_of_Classes)  #timm library
model_2 = efficientnet_b3(num_classes=Number_of_Classes)  #timm library


# loading trained model
state_dict_model = torch.load(address_of_trained_model_0)
#model.load_state_dict(state_dict_model)
from collections import OrderedDict
new_state_dict_model = OrderedDict()
for k, v in state_dict_model.items():
  name = k[7:] # remove `module.`
  new_state_dict_model[name] = v
model_0.load_state_dict(new_state_dict_model)


checkpoint_1 = torch.load(address_of_trained_model_1, map_location="cpu")
checkpoint_2 = torch.load(address_of_trained_model_2, map_location="cpu")
state_dict_pre_1 = checkpoint_1['state_dict']
state_dict_pre_2 = checkpoint_2['state_dict']


# rename moco pre-trained keys

state_dict = checkpoint_1['state_dict']
for k in list(state_dict.keys()):
  # retain only encoder_q up to before the embedding layer
  if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    # remove prefix
    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
  # delete renamed or unused k
  del state_dict[k]
msg = model_1.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}

state_dict = checkpoint_2['state_dict']
for k in list(state_dict.keys()):
  # retain only encoder_q up to before the embedding layer
  if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
    # remove prefix
    state_dict[k[len("module.encoder_q."):]] = state_dict[k]
  # delete renamed or unused k
  del state_dict[k]
msg = model_2.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}

model_0.cuda()
model_1.cuda()
model_2.cuda()
device = torch.device('cuda')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model_0 = torch.nn.DataParallel(model_0)
  model_1 = torch.nn.DataParallel(model_1)
  model_2 = torch.nn.DataParallel(model_2)
model_0.to(device)
model_1.to(device)
model_2.to(device)


torch.backends.cudnn.benchmark=True


N_train = XY_train.__len__()
N_val = XY_val.__len__()
N_unknown = XY_unknown.__len__()

feature_train = np.empty([N_train, N_feature+1])
feature_val = np.empty([N_val, N_feature+1])
feature_unknown = np.empty([N_unknown, N_feature+1])

model_0.eval()
model_1.eval()
model_2.eval()
print("start extracting feature in train datasets")
t1 = time.time()
n = 0
with torch.no_grad():
  for i, (x_supervised, x_moco,y) in enumerate(train_loader, 0):
    x_supervised = x_supervised.cuda()
    x_moco = x_moco.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV_0, Logit_0 = model_0(x_supervised)
    FV_1, Logit_1 = model_1(x_moco)
    FV_2, Logit_2 = model_2(x_moco)
    feature_train[n:(n+x_supervised.size(0)), 0] = y.cpu().data.numpy()
    feature_train[n:(n+x_supervised.size(0)), 1:] = np.hstack((FV_0.cpu().data.numpy(), FV_1.cpu().data.numpy(),FV_2.cpu().data.numpy()))
    n = n+x_supervised.size(0)
    
model_0.eval()
model_1.eval()
model_2.eval()  
print("start extracting feature in validation datasets")
n=0
with torch.no_grad():
  for i, (x_supervised, x_moco,y) in enumerate(val_loader, 0):
    x_supervised = x_supervised.cuda()
    x_moco = x_moco.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV_0, Logit_0 = model_0(x_supervised)
    FV_1, Logit_1 = model_1(x_moco)
    FV_2, Logit_2 = model_2(x_moco)
    feature_val[n:(n+x_supervised.size(0)), 0] = y.cpu().data.numpy()
    feature_val[n:(n+x_supervised.size(0)), 1:] = np.hstack((FV_0.cpu().data.numpy(), FV_1.cpu().data.numpy(),FV_2.cpu().data.numpy()))
    n = n+x_supervised.size(0)


model_0.eval()
model_1.eval()
model_2.eval()   
print("start extracting feature in unkown datasets")
n=0
with torch.no_grad():
  for i, (x_supervised, x_moco,y) in enumerate(unknown_loader, 0):
    x_supervised = x_supervised.cuda()
    x_moco = x_moco.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV_0, Logit_0 = model_0(x_supervised)
    FV_1, Logit_1 = model_1(x_moco)
    FV_2, Logit_2 = model_2(x_moco)
    feature_unknown[n:(n+x_supervised.size(0)), 0] = y.cpu().data.numpy()
    feature_unknown[n:(n+x_supervised.size(0)), 1:] = np.hstack((FV_0.cpu().data.numpy(), FV_1.cpu().data.numpy(),FV_2.cpu().data.numpy()))
    n = n+x_supervised.size(0)


t2 = time.time()
print("epoch time = ", t2-t1)

feature_train = feature_train[~np.isnan(feature_train).any(axis=1)]
feature_val = feature_val[~np.isnan(feature_val).any(axis=1)]
feature_unknown = feature_unknown[~np.isnan(feature_unknown).any(axis=1)]


np.save(file = ('/scratch/feature_train_' + model_name + '.npy'), arr=feature_train)
np.save(file = ('/scratch/feature_val_' + model_name + '.npy'), arr=feature_val)
np.save(file = ('/scratch/feature_unknown_' + model_name + '.npy'), arr=feature_unknown)

print('\nEnd')
