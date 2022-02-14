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
from torch import tensor
from timm.models.efficientnet import efficientnet_b3 as net_model_from_lib  # timm library
from my_lib import Average_Meter, Sum_Meter, accuracy

t0 = time.time()


number_of_classes = 1000

batch_size = 128
image_size = 300
max_number_of_epoch = 1000
learning_rate=0.00001
clipping_value = 256
feature_size = 1536 * 2
np.random.seed(2)
torch.manual_seed(2)

n_cpu = int(os.cpu_count()*0.8)



cnn_path_supervised = "/scratch/mjafarzadeh/trained_efficientnet_b3_fp16_imagenet.pth"
cnn_path_moco_places = "/scratch/mjafarzadeh/moco_places2_0199.pth"
train_csv_path = "/scratch/mjafarzadeh/imagenet_1000_train.csv"
val_csv_path = "/scratch/mjafarzadeh/imagenet_1000_val.csv"
assert os.path.isfile(cnn_path_supervised) 
assert os.path.isfile(cnn_path_moco_places) 
assert os.path.isfile(train_csv_path) 
assert os.path.isfile(val_csv_path) 



class csv_data_class_train(Dataset):

  def __init__(self, path, transform_common, transform_supervised, transform_moco):
    with open(path) as f:
      self.samples = [line.rstrip() for line in f if line != '']
    self.transform_common = transform_common
    self.transform_supervised = transform_supervised
    self.transform_moco = transform_moco

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    A, L = S.split(',')
    img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
    img_pil = PIL.Image.fromarray(img)
    x_common = self.transform_common(img_pil)
    x_supervised = self.transform_supervised(x_common)
    x_moco = self.transform_moco(x_common)
    y = int(L) - 1
    return (x_supervised, x_moco, y)




class csv_data_class_val(Dataset):

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
    y = int(L) - 1
    return (x_supervised, x_moco, y)


transform_train_common = transforms.Compose([
            transforms.RandomResizedCrop(size=(image_size,image_size), scale=(0.5, 2.0), ratio=(0.75, 1.25)),
            transforms.RandomHorizontalFlip(),
            transforms.AutoAugment(policy = transforms.autoaugment.AutoAugmentPolicy.IMAGENET, interpolation = transforms.InterpolationMode.BICUBIC, fill=127),
            transforms.ToTensor() ])

transform_train_supervised = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
transform_train_moco = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


transform_val_supervised = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])

transform_val_moco = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])


dataset_train = csv_data_class_train(path = train_csv_path,
                              transform_common = transform_train_common, 
                              transform_supervised = transform_train_supervised, 
                              transform_moco = transform_train_moco)
                              
dataset_val = csv_data_class_val(path = val_csv_path,
                              transform_supervised = transform_val_supervised, 
                              transform_moco = transform_val_moco)                 
                              
train_loader = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
val_loader = DataLoader(dataset=dataset_val, batch_size=batch_size, shuffle=False, num_workers=n_cpu)



t1 = time.time()

cnn_model_supervised = net_model_from_lib(num_classes = number_of_classes)  # timm library
cnn_model_moco_places = net_model_from_lib(num_classes = number_of_classes)  # timm library


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


t2 = time.time()
print("Loading cnn time = ", t2 - t1)




class Linear_Classifier(torch.nn.Module):
  def __init__(self, num_classes):
    super(Linear_Classifier, self).__init__()
    self.fc = torch.nn.Linear(feature_size, number_of_classes)

  def forward(self, x):
    return self.fc(x)

linear_model = Linear_Classifier(num_classes=number_of_classes) 
linear_model.cuda()


if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  cnn_model_supervised = torch.nn.DataParallel(cnn_model_supervised)
  cnn_model_moco_places = torch.nn.DataParallel(cnn_model_moco_places)
  linear_model = torch.nn.DataParallel(linear_model)


optimizer = torch.optim.Adam(linear_model.parameters(), lr=learning_rate)

best_val = 0.0

loss_CE = torch.nn.CrossEntropyLoss(reduction='sum').cuda()

torch.backends.cudnn.benchmark=True

loss_meter_train = Sum_Meter()
top1_train = Average_Meter()
top5_train = Average_Meter()
loss_meter_val = Sum_Meter()
top1_val = Average_Meter()
top5_val = Average_Meter()

for epoch in range(max_number_of_epoch):
  print("epoch = ", epoch +1)
  loss_meter_train.reset()
  top1_train.reset()
  top5_train.reset()
  loss_meter_val.reset()
  top1_val.reset()
  top5_val.reset()
  t3 = time.time()
  linear_model.train()
  for i, (x_supervised, x_moco, y) in enumerate(train_loader, 0):
    optimizer.zero_grad()
    with torch.no_grad():
      x_supervised = x_supervised.cuda()
      x_moco = x_moco.cuda()
      FV_supervised, Logit = cnn_model_supervised(x_supervised)
      FV_moco_places, Logit = cnn_model_moco_places(x_moco)
      FV = torch.cat((FV_supervised,  FV_moco_places), 1)
    y = y.long().cuda()
    Logit = linear_model(FV)
    with torch.no_grad():
      prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
    loss = loss_CE(Logit,y)
    with torch.no_grad():
      loss_meter_train.update(loss.item(), y.size(0))
      top1_train.update(prec1.item(), y.size(0))
      top5_train.update(prec5.item(), y.size(0))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(linear_model.parameters(), clipping_value)
    optimizer.step()
  print('train average_loss', loss_meter_train.avg)
  print('train top1 accuracy', top1_train.avg)
  print('train top5 accuracy ', top5_train.avg)
  linear_model.eval()
  with torch.no_grad():
    for i, (x_supervised, x_moco, y) in enumerate(val_loader, 0):
      x_supervised = x_supervised.cuda()
      x_moco = x_moco.cuda()
      FV_supervised, Logit = cnn_model_supervised(x_supervised)
      FV_moco_places, Logit = cnn_model_moco_places(x_moco)
      FV = torch.cat((FV_supervised,  FV_moco_places), 1)
      y = y.long().cuda()
      Logit = linear_model(FV)
      prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
      loss = loss_CE(Logit,y)
      loss_meter_val.update(loss.item(), y.size(0))
      top1_val.update(prec1.item(), y.size(0))
      top5_val.update(prec5.item(), y.size(0))
  t4 = time.time()
  print('validation average_loss', loss_meter_val.avg)
  print('validation top1 accuracy', top1_val.avg)
  print('validation top5 accuracy ', top5_val.avg)
  print("epoch time = ", t4-t3)
  if top1_val.avg > best_val:
    best_val = top1_val.avg
    print("model saved with vallidation top-1 accuracy  =  " , best_val)
    torch.save(linear_model.state_dict(), '/scratch/mjafarzadeh/linear_classifier_b3_SP_vall_acc_' + str(best_val) + '.pth')

print('Finished Training')
t5 = time.time()
print("Total time = ", t5-t0)
print('End')

