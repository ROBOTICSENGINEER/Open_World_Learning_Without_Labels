import torch
import numpy as np


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
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(1.0 / batch_size))
        return res




def sm_prediction(Logit):
  value, indices = torch.max(Logit, 1)
  return indices



def count_correct_wrong(Logit, Y_true):
  predicted = sm_prediction(Logit)
  is_wrong = (predicted != Y_true)
  is_correct = (predicted == Y_true)
  n_known_correct = torch.sum(is_correct)
  n_known_wrong = torch.sum(is_wrong)
  return (n_known_correct, n_known_wrong)


def CrossEntropy_loss_known_wrong(Logit, Y_true, predicted):
  is_known = (Y_true>=0)
  is_wrong = (predicted != Y_true)
  ind = is_known * is_wrong
  Logit_known_wrong = Logit[ind]
  Y_true_known_wrong = Y_true[ind]
  if Y_true_known_wrong.nelement() > 0:
    loss_CE = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    return loss_CE(Logit_known_wrong,Y_true_known_wrong)
  else:
    return torch.tensor(0.0).float().cuda()

def CrossEntropy_loss_knwon_correct(Logit, Y_true, predicted):
  is_known = (Y_true>=0) 
  is_correct = (predicted == Y_true)
  ind = is_known * is_correct
  Logit_known_correct = Logit[ind]
  Y_true_known_correct = Y_true[ind]
  if Y_true_known_correct.nelement() > 0:
    loss_CE = torch.nn.CrossEntropyLoss(reduction='sum').cuda()
    return loss_CE(Logit_known_correct,Y_true_known_correct)
  else:
    return torch.tensor(0.0).float().cuda()


def central_loss(x):
  return torch.sum( torch.clamp( 10.0 - torch.norm(x, p='fro', dim=1) , min=0.0) )



def CrossEntropy_Central_Loss(Logit, Y_true, alpha_correct  = 1.0 , alpha_wrong =1.0, alpha_central = 1.0):
  predicted = sm_prediction(Logit)
  LKW_CE = alpha_wrong * CrossEntropy_loss_known_wrong(Logit=Logit, Y_true=Y_true, predicted = predicted)
  LKC_CE = alpha_correct * CrossEntropy_loss_knwon_correct(Logit=Logit, Y_true=Y_true, predicted = predicted)
  LCL    = alpha_central * central_loss(Logit)
  loss = LKW_CE + LKC_CE  + LCL
  return loss


  
# def MSE(p,y):
#   # print("y = ", y)
#   q = torch.nn.functional.one_hot(y,num_classes = p.shape[1]).float().cuda()
#   # print("p = ", p)
#   # print("q = ", q)
#   # print("p.shape = ", p.shape)
#   # print("q.shape = ", q.shape)
#   return torch.nn.functional.mse_loss(p,q, reduction = 'sum')
