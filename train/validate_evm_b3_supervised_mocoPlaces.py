import time
import numpy as np
import pickle
import cv2
import PIL
import torch
import os
import json
import argparse

from collections import OrderedDict
from functools import partial

from torch import Tensor

import torch.multiprocessing as mp

from my_lib import *

from EVM import EVM_Training , EVM_Inference


import weibull
import sys
sys.modules['weibull'] = weibull

from statistics import mean
import gc

try:
  torch.multiprocessing.set_sharing_strategy('file_system')
except RuntimeError:
  pass


try:
  mp.set_start_method('spawn', force=True)
except RuntimeError:
  pass


number_of_classes = 1000


n_cpu = int(os.cpu_count()*0.8)


def val_process(classes_to_process, feature_dic, evm_model, args_evm, gpu, Q, done_event):
  with torch.no_grad():
    top1_Meter = Average_Meter()
    Pr_iterator = EVM_Inference(classes_to_process, feature_dic, args_evm, gpu, evm_model)

    for k,pr in enumerate(Pr_iterator):
      r = pr[1][1].cuda(gpu)
      m, m_i  = torch.max(r, dim = 1)
      u = (1 - m).view(-1, 1)
      q = torch.cat((u, r), 1)
      norm = torch.norm(q, p=1, dim=1)
      p = q/norm[:,None]
      L = ( (pr[1][0]) * torch.ones(r.shape[0])).long().cuda(gpu)
      acc = accuracy(p, L, topk=(1, ))
      top1_Meter.update(acc[0].item(), r.size(0))
    Q.put((gpu, top1_Meter.avg, top1_Meter.count))
    done_event.wait()
    del r, m, m_i, u, q, norm, p
    del L, acc, top1_Meter, Pr_iterator


def val_process_unknown(classes_to_process, feature_dic, evm_model, args_evm, gpu, Q, done_event):
  with torch.no_grad():
    top1_Meter = Average_Meter()
    Pr_iterator = EVM_Inference(classes_to_process, feature_dic, args_evm, gpu, evm_model)

    for k,pr in enumerate(Pr_iterator):
      r = pr[1][1].cuda(gpu)
      m, m_i  = torch.max(r, dim = 1)
      u = (1 - m).view(-1, 1)
      q = torch.cat((u, r), 1)
      _, y = torch.max(1, dim = 1)
      acc = torch.sum(y==0) / q.shape[0]
      top1_Meter.update(acc[0].item(), r.size(0))
    Q.put((gpu, top1_Meter.avg, top1_Meter.count))
    done_event.wait()
    del r, m, m_i, u, q, y
    del acc, top1_Meter, Pr_iterator

def validate(evm_model, args_evm, class_partition, feature_dic, data_name, gpu):
  with torch.no_grad():
    print(f"start evaluating {data_name}")
    t1 = time.time()
    NG = min(  len(gpu) , number_of_classes  )
    assert NG > 0
    classes_to_process = list(range(1,int(number_of_classes)+1))
    list_acc = [0.0] * NG
    list_count = [0] * NG
    Q = mp.Queue()
    done_event = [mp.Event() for k in range(NG)]

    process_list = []
    if data_name != "unknown":
      for k in range(NG):
        p = mp.Process(target=val_process, args=(class_partition[k], feature_dic[k], evm_model, args_evm, gpu[k], Q, done_event[k]))
        p.start()
        process_list.append(p)
    else:
      for k in range(NG):
        p = mp.Process(target=val_process_unknown, args=(class_partition[k], feature_dic[k], evm_model, args_evm, gpu[k], Q, done_event[k]))
        p.start()
        process_list.append(p)
    
    for k in range(NG):
      g, a , c = Q.get()
      print(g,a,c)
      i = gpu.index(g)
      list_acc[i] = a
      list_count[i] = c
      done_event[i].set()
      
    for p in process_list:
      p.join()
      
    print(data_name, "total accuracy = ", np.average(np.array(list_acc), weights=np.array(list_count)))

    del Q, done_event
    del p, process_list, g, a , c, list_acc, list_count
    t2 = time.time()
    print("validation time = ", t2 - t1)
    return


########################################
########################################
########################################


if __name__ == '__main__':
  
  print(f"Start")
  t0 = time.time()
  
  # with open('evm_config_mini_imagenet.json', 'r') as json_file:
  with open('evm_config_cosine.json', 'r') as json_file:
    evm_config = json.loads(json_file.read())
  cover_threshold = evm_config['cover_threshold']
  distance_multiplier = evm_config['distance_multiplier']
  tail_size = evm_config['tail_size']
  distance_metric =  evm_config['distance_metric']

  
  torch.backends.cudnn.benchmark=True
  
  with torch.no_grad():
    args_evm  = argparse.Namespace()
    args_evm.cover_threshold = [cover_threshold]
    args_evm.distance_multiplier = [distance_multiplier]
    args_evm.tailsize = [tail_size]
    args_evm.distance_metric = distance_metric
    args_evm.chunk_size = 200
  
  
    gpu_count = torch.cuda.device_count()
    list_of_all_gpu = list(range(gpu_count))
    NG = len(list_of_all_gpu)
    
    filename = "/scratch/mjafarzadeh/evm_cosine_imagenet_b3_joint_supervised_mocoPlaces_tail33998_ct7_dm45.pkl"
  
    evm_model = pickle.load( open(filename , "rb" ) )
    
    t1 = time.time()
    print(f"loading evm time = {t1-t0}")
    '''
    data_list = [ ('val', '/scratch/mjafarzadeh/feature_b3_SP_val.pth'), 
                  ('unknown', '/scratch/mjafarzadeh/feature_b3_SP_unknown.pth'), 
                  ('test_hard', '/scratch/mjafarzadeh/feature_b3_SP_test_hard.pth' ), 
                  ('test_medium' , '/scratch/mjafarzadeh/feature_b3_SP_test_medium.pth'), 
                  ('test_easy' , '/scratch/mjafarzadeh/feature_b3_SP_test_easy.pth')]
    '''
    data_list = [ ('val', '/scratch/mjafarzadeh/feature_b3_SP_val.pth'),  
                  ('test_hard', '/scratch/mjafarzadeh/feature_b3_SP_test_hard.pth' ), 
                  ('test_medium' , '/scratch/mjafarzadeh/feature_b3_SP_test_medium.pth'), 
                  ('test_easy' , '/scratch/mjafarzadeh/feature_b3_SP_test_easy.pth')]
  
    for data_name, data_path in data_list:
      t2 = time.time()
      
      print(f"\nstart {data_name}")
      
      if data_path[-1] == 'y':
        data =  torch.from_numpy(np.load(data_path))
      else:
        data =  torch.load(data_path)
  
      t3 = time.time()
      print(f"loading {data_name} feature time = {t3-t2}")
    
      features_dict_val = [OrderedDict() for k in range(NG)]
      class_partition = [[] for k in range(NG)]
      
      if data_name == "unknown":
        k1 = int(torch.min(data[:,0]))
        k2 = int(torch.max(data[:,0]))
        
        for k in range(k1, k2+1):
          F = data[data[:,0]==k]
          r = k % NG
          class_partition[r].append(k)
          (features_dict_val[r])[k] = F[:,1:].detach().clone().double()
      else:
        for k in range(1, number_of_classes+1):
          F = data[data[:,0]==k]
          r = k % NG
          class_partition[r].append(k)
          (features_dict_val[r])[k] = F[:,1:].detach().clone().double()
      
      t4 = time.time()
      print(f"computing features_dict {data_name} time = {t4-t3}")
    
      validate(evm_model, args_evm, class_partition, features_dict_val, data_name , list_of_all_gpu)
    
      t5 = time.time()
      print(f"valiation of val data time = {t5-t4}")
      
    #print("validation accuracy = " , acc)
    t6 = time.time()
    print(f"run time = {t6-t0}")
    


