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

from timm.models.efficientnet import efficientnet_b3  # timm library


model_name = 'efficientnet_b3_fp16_imagenet'
address_of_trained_model = '/scratch/trained_efficientnet_b3_fp16_imagenet.pth.tar'

n_cpu = 32
batch_size = 256
LR = 0.001
image_size = 300
N_feature = 1536
Number_of_Classes = 1000

np.random.seed(2)
torch.manual_seed(2)

t1 = time.time()


class Average_Meter(object):
  """Computes and stores the average and current value"""
  def __init__(self):
    self.reset()

  def reset(self):
    self.val = 0.0
    self.avg = 0.0
    self.sum = 0.0
    self.count = 0

  def update(self, val, n):
    if n > 0:
      self.val = val
      self.sum += val * n
      self.count += n
      self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


class known_train_data_class(Dataset):

  def __init__(self, transform=None):
    with open('./data/imagenet_1000_train.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x = self.transform(img_pil)
      y = int(L)-1
    return (x,y)


class known_val_data_class(Dataset):

  def __init__(self, transform=None):
    with open('./data/imagenet_1000_val.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x = self.transform(img_pil)
      y = int(L)-1
    return (x,y)



class unknown_train_data_class(Dataset):

  def __init__(self, transform=None):
    with open('./data/imagenet_166.csv') as f:
      self.samples = [line.rstrip() for line in f if line is not '']
    self.transform = transform

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    if type(index) == type(int(1)):
      A, L = S.split(',')
      img = cv2.cvtColor(cv2.imread(A,1), cv2.COLOR_BGR2RGB)
      img_pil = PIL.Image.fromarray(img)
      x = self.transform(img_pil)
      y = int(L)-1
    return (x,y)


image_transform_val = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])

XY_train = known_train_data_class(transform = image_transform_val)
XY_val = known_val_data_class(transform = image_transform_val)
XY_unknown = unknown_train_data_class(transform = image_transform_val)

train_loader = DataLoader(dataset=XY_train, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
val_loader = DataLoader(dataset=XY_val, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
unknown_loader = DataLoader(dataset=XY_unknown, batch_size=batch_size, shuffle=False, num_workers=n_cpu)


model = efficientnet_b3(num_classes=Number_of_Classes)  #timm library

# loading trained model
state_dict_model = torch.load(address_of_trained_model)
#model.load_state_dict(state_dict_model)
from collections import OrderedDict
new_state_dict_model = OrderedDict()
for k, v in state_dict_model.items():
  name = k[7:] # remove `module.`
  new_state_dict_model[name] = v
model.load_state_dict(new_state_dict_model)


  
  
model.cuda()

optimizer = torch.optim.Adam(model.parameters(), lr=LR)

device = torch.device('cuda')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model.to(device)


def sm_prediction(logit):
  value, indices = torch.max(logit, 1)
  return indices


torch.backends.cudnn.benchmark=True

#print(model)
summary(model, ( 3, image_size, image_size), batch_size=-1, device='cuda')



top1_train = Average_Meter()
top5_train = Average_Meter()
top1_val = Average_Meter()
top5_val = Average_Meter()


top1_train.reset()
top5_train.reset()
top1_val.reset()
top5_val.reset()

N_train = XY_train.__len__()
N_val = XY_val.__len__()
N_unknown = XY_unknown.__len__()

feature_train = np.empty([N_train, N_feature+1])
feature_val = np.empty([N_val, N_feature+1])
feature_unknown = np.empty([N_unknown, N_feature+1])



model.eval()
print("start extracting feature in train datasets")
n = 0
with torch.no_grad():
  for i, (x,y) in enumerate(train_loader, 0):
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV, Logit = model(x)
    prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
    predicted = sm_prediction(Logit)
    top1_train.update(prec1.item(), x.size(0))
    top5_train.update(prec5.item(), x.size(0))
    feature_train[n:(n+x.size(0)), 0] = y.cpu().data.numpy()
    feature_train[n:(n+x.size(0)), 1:] = FV.cpu().data.numpy()
    n = n+x.size(0)
    
model.eval()    
print("start extracting feature in validation datasets")
n=0
with torch.no_grad():
  for i, (x,y) in enumerate(val_loader, 0):
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV, Logit = model(x)
    prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
    predicted = sm_prediction(Logit)
    top1_val.update(prec1.item(), x.size(0))
    top5_val.update(prec5.item(), x.size(0))
    feature_val[n:(n+x.size(0)), 0] = y.cpu().data.numpy()
    feature_val[n:(n+x.size(0)), 1:] = FV.cpu().data.numpy()
    n = n+x.size(0)

model.eval()    
print("start extracting feature in unknown datasets")
n=0
with torch.no_grad():
  for i, (x,y) in enumerate(unknown_loader, 0):
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV, Logit = model(x)
    feature_unknown[n:(n+x.size(0)), 0] = y.cpu().data.numpy()
    feature_unknown[n:(n+x.size(0)), 1:] = FV.cpu().data.numpy()
    n = n+x.size(0)


t2 = time.time()
print('train top1 accuracy', top1_train.avg)
print('train top5 accuracy ', top5_train.avg)
print('validation top1 accuracy', top1_val.avg)
print('validation top5 accuracy ', top5_val.avg)
print("total time = ", t2-t1)


feature_train = feature_train[~np.isnan(feature_train).any(axis=1)]
feature_val = feature_val[~np.isnan(feature_val).any(axis=1)]
feature_unknown = feature_unknown[~np.isnan(feature_unknown).any(axis=1)]

np.save(file = ('/scratch/feature_train_' + model_name + '.npy'), arr=feature_train)
np.save(file = ('/scratch/feature_val_' + model_name + '.npy'), arr=feature_val)
np.save(file = ('/scratch/feature_unknown_' + model_name + '.npy'), arr=feature_unknown)

print('\nEnd')
