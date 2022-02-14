import numpy as np
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
from torch.cuda.amp import autocast, GradScaler

n_cpu = 32
batch_size = 256
max_number_of_epoch = 1000
LR = 0.00001
alpha_wrong = 1.0
image_size = 300
number_of_classes = 1000

clipping_value = 512

np.random.seed(2)
torch.manual_seed(2)


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


class Sum_Meter(object):
  """Computes and stores the sum and current value"""
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
      self.sum += val
      self.count += n
      self.avg = self.sum / self.count



def accuracy(output, target, topk=(1,)):
  with torch.no_grad():
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
      correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
      res.append(correct_k.mul_(100.0 / batch_size))
    return res


class known_train_data_class(Dataset):

  def __init__(self, transform=None):
    with open('./preparing/imagenet_1000_train.csv') as f:
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
    with open('./preparing/imagenet_1000_val.csv') as f:
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


image_transform_train = transforms.Compose([
            transforms.RandomAffine(degrees=45, translate=(0.25, 0.25), scale=(0.5, 4.0), shear=None, resample=PIL.Image.BICUBIC, fillcolor=127),
            transforms.RandomResizedCrop(size=(image_size,image_size), scale=(0.1, 1.0), ratio=(0.5, 1.5)), transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.4)], p=0.5),
            transforms.RandomGrayscale(p=0.1),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)) ])


image_transform_val = transforms.Compose([
            transforms.Resize(size=(image_size,image_size), interpolation=PIL.Image.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])

XY_train = known_train_data_class(transform = image_transform_train)
XY_val = known_val_data_class(transform = image_transform_val)

train_loader = DataLoader(dataset=XY_train, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
val_loader = DataLoader(dataset=XY_val, batch_size=batch_size, shuffle=True, num_workers=n_cpu)



model = efficientnet_b3(num_classes=number_of_classes)  #timm library
best_val = 0.0



best_val = 79.618
# loading trained original model
# model.load_state_dict(torch.load('/scratch/mjafarzadeh/cvpr/tf_efficientnet_b3_ns-9d44bf68.pth'))

state_dict_model = torch.load('/scratch/mjafarzadeh/trained_efficientnet_b3_fp16_imagenet.pth.tar')
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


def count_correct_wrong(logit, Y_true):
  predicted = sm_prediction(logit)
  is_wrong = (predicted != Y_true)
  is_correct = (predicted == Y_true)
  n_known_correct = torch.sum(is_correct)
  n_known_wrong = torch.sum(is_wrong)
  return (n_known_correct, n_known_wrong)




def CrossEntropy_loss_known_wrong(logit, Y_true, predicted):
  is_known = (Y_true>=0)
  is_wrong = (predicted != Y_true)
  ind = is_known * is_wrong
  logit_known_wrong = logit[ind]
  Y_true_known_wrong = Y_true[ind]
  if Y_true_known_wrong.nelement() > 0:
    loss_CE = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    return loss_CE(logit_known_wrong,Y_true_known_wrong)
  else:
    return torch.tensor(0.0).float().cuda()

def CrossEntropy_loss_known_correct(logit, Y_true, predicted):
  is_known = (Y_true>=0) 
  is_correct = (predicted == Y_true)
  ind = is_known * is_correct
  logit_known_correct = logit[ind]
  Y_true_known_correct = Y_true[ind]
  if Y_true_known_correct.nelement() > 0:
    loss_CE = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    return loss_CE(logit_known_correct,Y_true_known_correct)
  else:
    return torch.tensor(0.0).float().cuda()


torch.backends.cudnn.benchmark=True

#print(model)
summary(model, ( 3, image_size, image_size), batch_size=-1, device='cuda')


losses_train_wrong = Sum_Meter()
losses_train_correct = Sum_Meter()
losses_train_total = Sum_Meter()
top1_train = Average_Meter()
top5_train = Average_Meter()
losses_val_wrong = Sum_Meter()
losses_val_correct = Sum_Meter()
losses_val_total = Sum_Meter()
top1_val = Average_Meter()
top5_val = Average_Meter()


# Creates a GradScaler once at the beginning of training.
scaler = GradScaler()

for epoch in range(max_number_of_epoch):
  print("epoch = ", epoch +1)
  losses_train_wrong.reset()
  losses_train_correct.reset()
  losses_train_total.reset()
  top1_train.reset()
  top5_train.reset()
  losses_val_wrong.reset()
  losses_val_correct.reset()
  losses_val_total.reset()
  top1_val.reset()
  top5_val.reset()
  t1 = time.time()
  model.train()
  for i, (x,y) in enumerate(train_loader, 0):
    #print("train iteration   " , i)
    optimizer.zero_grad()
    with autocast():
      x = x.half().cuda()
      y = tensor(y, dtype=torch.int64).cuda()
      FV, Logit = model(x)
      prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
      predicted = sm_prediction(Logit)
      LKW_CE = alpha_wrong * CrossEntropy_loss_known_wrong(logit=Logit, Y_true=y, predicted = predicted)
      LKC_CE = CrossEntropy_loss_known_correct(logit=Logit, Y_true=y, predicted = predicted)
      loss = LKW_CE + LKC_CE
      n_known_correct, n_known_wrong = count_correct_wrong(Logit, y)
      losses_train_wrong.update(LKW_CE.item(), n_known_wrong.item())
      losses_train_correct.update(LKC_CE.item(), n_known_correct.item())
      losses_train_total.update(loss.item(), y.size(0))
      top1_train.update(prec1.item(), y.size(0))
      top5_train.update(prec5.item(), y.size(0))
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    torch.nn.utils.clip_grad_norm_(model.parameters(), clipping_value)
    scaler.step()
  model.eval()
  with torch.no_grad():
    for i, (x,y) in enumerate(val_loader, 0):
      with autocast():
        x = x.cuda()
        y = tensor(y, dtype=torch.int64).cuda()
        FV, Logit = model(x)
        prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
        predicted = sm_prediction(Logit)
        LKW_CE = alpha_wrong * CrossEntropy_loss_known_wrong(logit=Logit, Y_true=y, predicted = predicted)
        LKC_CE = CrossEntropy_loss_known_correct(logit=Logit, Y_true=y, predicted = predicted)
        loss = LKW_CE + LKC_CE
        n_known_correct, n_known_wrong = count_correct_wrong(Logit, y)
        losses_val_wrong.update(LKW_CE.item(), n_known_wrong.item())
        losses_val_correct.update(LKC_CE.item(), n_known_correct.item())
        losses_val_total.update(loss.item(), y.size(0))
        top1_val.update(prec1.item(), y.size(0))
        top5_val.update(prec5.item(), y.size(0))
  t2 = time.time()
  print('train number_wrong', losses_train_wrong.count)
  print('train number_correct', losses_train_correct.count)
  print('train average_loss_wrong', losses_train_wrong.avg)
  print('train average_loss_correct', losses_train_correct.avg)
  print('train average_loss_total', losses_train_total.avg)
  print('train top1 accuracy', top1_train.avg)
  print('train top5 accuracy ', top5_train.avg)
  print('validation number_wrong', losses_val_wrong.count)
  print('validation number_correct', losses_val_correct.count) 
  print('validation average_loss_wrong', losses_val_wrong.avg)
  print('validation average_loss_correct', losses_val_correct.avg)
  print('validation average_loss_total', losses_val_total.avg)
  print('validation top1 accuracy', top1_val.avg)
  print('validation top5 accuracy ', top5_val.avg)
  print("epoch time = ", t2-t1)
  if top1_val.avg > best_val:
    best_val = top1_val.avg
    print("model saved with vallidation top-1 accuracy  =  " , best_val)
    torch.save(model.state_dict(), 'efficientnet_b3_fp16_imagenet_vall_acc_'+str(best_val)+ '.pth.tar')

print('Finished Training')
