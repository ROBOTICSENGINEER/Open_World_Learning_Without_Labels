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
from EVM import EVM_Training, EVM_Inference

import weibull
import sys

sys.modules["weibull"] = weibull

t0 = time.time()


feature_path = "/scratch/mjafarzadeh/feature_b3_SP_val.pth"
evm_path = "/scratch/mjafarzadeh/evm_cosine_imagenet_b3_joint_supervised_mocoPlaces_tail33998_ct7_dm45.pkl"
linear_path = "/scratch/mjafarzadeh/linear_classifier_b3_SP_vall_acc_0.8148800000190735.pth"
batch_size = 1000
number_of_known_classes = 1000
number_of_classes = 1000
feature_size = 1536 * 2
cover_threshold = 0.7
distance_multiplier = 0.45
tail_size = 33998
distance_metric = "cosine"
chunk_size = 100
n_cpu = int(os.cpu_count() * 0.8)
image_size = 300
np.random.seed(2)
torch.manual_seed(2)

assert os.path.isfile(feature_path)
assert os.path.isfile(linear_path)
assert os.path.isfile(evm_path)

t1 = time.time()

N = 50000
data = torch.load(feature_path)[:, 1:]
data = data[~torch.any(data.isnan(), dim=1)]
print(data.shape[0])
assert data.shape[0] == N
data.cuda()

t2 = time.time()
print("loading feature time = ", t2 - t1)

SoftMax = torch.nn.Softmax(dim=1).cuda()


class Linear_Classifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(Linear_Classifier, self).__init__()
        self.fc = torch.nn.Linear(feature_size, number_of_classes)

    def forward(self, x):
        return self.fc(x)


linear_model = Linear_Classifier(num_classes=number_of_classes)

assert os.path.isfile(linear_path)
checkpoint = torch.load(linear_path)
if "epoch" in checkpoint.keys():
    state_dict_model = checkpoint["state_dict"]
else:
    state_dict_model = checkpoint
from collections import OrderedDict

new_state_dict_model = OrderedDict()
for k, v in state_dict_model.items():
    if "module." == k[:7]:
        name = k[7:]  # remove `module.`
        new_state_dict_model[name] = v
    else:
        new_state_dict_model[k] = v
linear_model.load_state_dict(new_state_dict_model)
for parameter in linear_model.parameters():
    parameter.requires_grad = False
linear_model.cuda()
linear_model.eval()

if torch.cuda.device_count() > 1:
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    linear_model = torch.nn.DataParallel(linear_model)


t3 = time.time()
print("loading linear model time = ", t3 - t2)


args_evm = argparse.Namespace()
args_evm.cover_threshold = [cover_threshold]
args_evm.distance_multiplier = [distance_multiplier]
args_evm.tailsize = [tail_size]
args_evm.distance_metric = distance_metric
args_evm.chunk_size = chunk_size


torch.backends.cudnn.benchmark = True
with torch.no_grad():
    evm_model = pickle.load(open(evm_path, "rb"))

    t4 = time.time()
    print("Loading evm time = ", t4 - t3)

    storage = torch.empty(N, 3)
    print(storage.shape)

    for i1 in range(0, N, batch_size):
        i2 = i1 + batch_size
        FV = data[i1:i2, :]

        Logit = linear_model(FV)
        negatvie_energy = torch.logsumexp(Logit, dim=1)
        softmax = SoftMax(Logit)
        sm, _ = torch.max(softmax, axis=1)

        feature_dict = OrderedDict()
        feature_dict[0] = FV.double()
        Pr_iterator = EVM_Inference([0], feature_dict, args_evm, 0, evm_model)
        for j, pr in enumerate(Pr_iterator):
            prob = pr[1][1]  # .cuda()
            assert j == 0
        del Pr_iterator, pr
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        p_evm, _ = torch.max(prob, axis=1)

        storage[i1:i2, 0] = sm.cpu()
        storage[i1:i2, 1] = negatvie_energy.cpu()
        storage[i1:i2, 2] = p_evm.cpu()

    sm = storage[:, 0]
    negatvie_energy = storage[:, 1]
    p = storage[:, 2]

    sm, _ = torch.sort(sm)
    negatvie_energy, _ = torch.sort(negatvie_energy)
    p, _ = torch.sort(p)

    k = int(0.05 * N) - 1

    thresh_sm = sm[k].item()
    thresh_energy = negatvie_energy[k].item()
    thresh_evm = p[k].item()

    print(f"{thresh_sm = }")
    print(f"{thresh_energy = }")
    print(f"{thresh_evm = }")


with open("threshold_b3_sp.txt", "w") as f:
    f.write(f"Efficientnet B3 SP\n")
    f.write(f"thresh_sm = {thresh_sm}\n")
    f.write(f"thresh_energy = {thresh_energy}\n")
    f.write(f"thresh_evm = {thresh_evm}\n")
