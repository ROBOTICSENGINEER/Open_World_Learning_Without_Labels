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

#model_name = 'moco_b3_imagenet'
#address_of_trained_model = '/scratch/moco_imagenet_0199.pth.tar'

model_name = 'moco_b3_places2'
address_of_trained_model = '/scratch/moco_places2_0199.pth.tar'


n_cpu = 32
batch_size = 512
image_size = 300
N_feature = 1536
Number_of_Classes = 1000

np.random.seed(2)
torch.manual_seed(2)

t1 = time.time()

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
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

XY_train = known_train_data_class(transform = image_transform_val)
XY_val = known_val_data_class(transform = image_transform_val)
XY_unknown = unknown_train_data_class(transform = image_transform_val)

train_loader = DataLoader(dataset=XY_train, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
val_loader = DataLoader(dataset=XY_val, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
unknown_loader = DataLoader(dataset=XY_unknown, batch_size=batch_size, shuffle=False, num_workers=n_cpu)

model = efficientnet_b3(num_classes=Number_of_Classes)  #timm library
checkpoint = torch.load(address_of_trained_model, map_location="cpu")
state_dict_pre = checkpoint['state_dict']

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

msg = model.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}

model.cuda()
device = torch.device('cuda')
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda.device_count(), "GPUs!")
  model = torch.nn.DataParallel(model)
model.to(device)


torch.backends.cudnn.benchmark=True

#print(model)
#summary(model, ( 3, image_size, image_size), batch_size=-1, device='cuda')


N_train = XY_train.__len__()
N_val = XY_val.__len__()
N_unknown = XY_unknown.__len__()

feature_train = np.empty([N_train, N_feature+1])
feature_val = np.empty([N_val, N_feature+1])
feature_unknown = np.empty([N_unknown, N_feature+1])



model.eval()
print("start extracting feature in train datasets")
t1 = time.time()
n = 0
with torch.no_grad():
  for i, (x,y) in enumerate(train_loader, 0):
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    FV, Logit = model(x)
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
    feature_val[n:(n+x.size(0)), 0] = y.cpu().data.numpy()
    feature_val[n:(n+x.size(0)), 1:] = FV.cpu().data.numpy()
    n = n+x.size(0)


model.eval()    
print("start extracting feature in unkown datasets")
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
print("epoch time = ", t2-t1)


feature_train = feature_train[~np.isnan(feature_train).any(axis=1)]
feature_val = feature_val[~np.isnan(feature_val).any(axis=1)]
feature_unknown = feature_unknown[~np.isnan(feature_unknown).any(axis=1)]


np.save(file = ('/scratch/feature_train_' + model_name + '.npy'), arr=feature_train)
np.save(file = ('/scratch/feature_val_' + model_name + '.npy'), arr=feature_val)
np.save(file = ('/scratch/feature_unknown_' + model_name + '.npy'), arr=feature_unknown)

print('\nEnd')
