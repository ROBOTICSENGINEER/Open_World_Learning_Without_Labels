import numpy as np
from random import shuffle
import torch
from torchsummary import summary
from torch import from_numpy, tensor
from torch.utils.data import Dataset, DataLoader
import time



n_cpu = 32
batch_size = 16384
max_number_of_epoch = 1000
LR = 0.01
alpha_wrong = 1.0
number_of_classes = 1000

np.random.seed(2)
torch.manual_seed(2)


'''
model_name = "moco_b3_imagenet"
address_feature_train = "/scratch/feature_train_moco_b3_imagenet.npy"
address_feature_val = "/scratch/feature_val_moco_b3_imagenet.npy"

'''
model_name = "moco_b3_places2"
address_feature_train = "/scratch/feature_train_moco_b3_places2.npy"
address_feature_val = "/scratch/feature_val_moco_b3_places2.npy"




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
    self.samples = np.load(address_feature_train).astype(np.float32)

  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, index):
    x = self.samples[index,1:]
    y = self.samples[index,0]
    return (x,y)


class known_val_data_class(Dataset):
  
  def __init__(self, transform=None):
    self.samples = np.load(address_feature_val).astype(np.float32)

  def __len__(self):
    return self.samples.shape[0]

  def __getitem__(self, index):
    x = self.samples[index,1:]
    y = self.samples[index,0]
    return (x,y)


XY_train = known_train_data_class()
XY_val = known_val_data_class()

train_loader = DataLoader(dataset=XY_train, batch_size=batch_size, shuffle=True, num_workers=n_cpu)
val_loader = DataLoader(dataset=XY_val, batch_size=batch_size, shuffle=True, num_workers=n_cpu)


class Linear_Classifier(torch.nn.Module):
  def __init__(self, num_classes):
    super(Linear_Classifier, self).__init__()
    self.fc = torch.nn.Linear(1536, num_classes)

  def forward(self, x):
    return self.fc(x)




model = Linear_Classifier(num_classes=number_of_classes) 
best_val = 0.0

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

def CrossEntropy_loss_knwon_correct(logit, Y_true, predicted):
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
    x = x.cuda()
    y = tensor(y, dtype=torch.int64).cuda()
    Logit = model(x)
    prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
    predicted = sm_prediction(Logit)
    LKW_CE = alpha_wrong * CrossEntropy_loss_known_wrong(logit=Logit, Y_true=y, predicted = predicted)
    LKC_CE = CrossEntropy_loss_knwon_correct(logit=Logit, Y_true=y, predicted = predicted)
    loss = LKW_CE + LKC_CE
    n_known_correct, n_known_wrong = count_correct_wrong(Logit, y)
    losses_train_wrong.update(LKW_CE.item(), n_known_wrong.item())
    losses_train_correct.update(LKC_CE.item(), n_known_correct.item())
    losses_train_total.update(loss.item(), y.size(0))
    top1_train.update(prec1.item(), y.size(0))
    top5_train.update(prec5.item(), y.size(0))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 512)
    optimizer.step()
  model.eval()
  with torch.no_grad():
    for i, (x,y) in enumerate(val_loader, 0):
      x = x.cuda()
      y = tensor(y, dtype=torch.int64).cuda()
      Logit = model(x)
      prec1, prec5 = accuracy(Logit.data, y, topk=(1, 5))
      predicted = sm_prediction(Logit)
      LKW_CE = alpha_wrong * CrossEntropy_loss_known_wrong(logit=Logit, Y_true=y, predicted = predicted)
      LKC_CE = CrossEntropy_loss_knwon_correct(logit=Logit, Y_true=y, predicted = predicted)
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
    torch.save(model.state_dict(), 'linear_classifier_' + model_name + '_vall_acc_' + str(best_val) + '.pth.tar')

print('Finished Training')
