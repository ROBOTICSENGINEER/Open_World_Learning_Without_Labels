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
from EVM import EVM_Training, EVM_Inference


import weibull
import sys

sys.modules["weibull"] = weibull


t0 = time.time()


"""
data_list = [ ('train', '/scratch/mjafarzadeh/feature_train_b3_joint_supervised_mocoPlaces.npy'),
              ('val', '/scratch/mjafarzadeh/feature_b3_SP_val.pth'),
              ('test_hard', '/scratch/mjafarzadeh/feature_b3_SP_test_hard.pth' ),
              ('test_medium' , '/scratch/mjafarzadeh/feature_b3_SP_test_medium.pth'),
              ('test_easy' , '/scratch/mjafarzadeh/feature_b3_SP_test_easy.pth')]
"""

data_list = [
    ("val", "/scratch/mjafarzadeh/feature_b3_SP_val.pth"),
    ("test_hard", "/scratch/mjafarzadeh/feature_b3_SP_test_hard.pth"),
    ("test_medium", "/scratch/mjafarzadeh/feature_b3_SP_test_medium.pth"),
    ("test_easy", "/scratch/mjafarzadeh/feature_b3_SP_test_easy.pth"),
]


config = json.load(open("config_owl_b3_supervised_mocoPlaces.json"))

batch_size = 1000


number_of_known_classes = config["number_of_known_classes"]
number_of_classes = config["number_of_known_classes"]
evm_path = config["evm_path"]
cover_threshold = config["cover_threshold"]
distance_multiplier = config["distance_multiplier"]
tail_size = config["tail_size"]
distance_metric = config["distance_metric"]
chunk_size = config["chunk_size"]
np.random.seed(config["random_seed"])
torch.manual_seed(config["random_seed"])


torch.backends.cudnn.benchmark = True
t1 = time.time()
assert os.path.isfile(evm_path)

args_evm = argparse.Namespace()
args_evm.cover_threshold = [cover_threshold]
args_evm.distance_multiplier = [distance_multiplier]
args_evm.tailsize = [tail_size]
args_evm.distance_metric = distance_metric
args_evm.chunk_size = chunk_size


with torch.no_grad():
    evm_model = pickle.load(open(evm_path, "rb"))

    t2 = time.time()
    print("Loading evm time = ", t2 - t1)

    for data_name, data_path in data_list:
        t3 = time.time()

        print(f"\nstart {data_name}")

        if data_path[-1] == "y":
            data = torch.from_numpy(np.load(data_path))
        else:
            data = torch.load(data_path)
        print(f"{data.shape = }")
        data = data[~torch.any(data.isnan(), dim=1)]
        FV = data[:, 1:]
        L = data[:, 0]
        N = data.shape[0]
        if (torch.min(L) == 0) and (torch.max(L) == 999):
            L = L + 1
            data[:, 0] = L

        assert torch.min(L) == 1
        assert torch.max(L) == 1000

        print(f"{data.shape = }")

        t4 = time.time()
        print(f"loading {data_name} feature time = {t4-t3}")

        for k in range(0, N, batch_size):
            k1 = k
            k2 = min(N, k + batch_size)
            feature_dict = OrderedDict()
            feature_dict[0] = FV[k1:k2, :].double()
            y = L[k1:k2]
            Pr_iterator = EVM_Inference([0], feature_dict, args_evm, 0, evm_model)
            for j, pr in enumerate(Pr_iterator):
                prob = pr[1][1]  # .cuda()
                assert j == 0
            del Pr_iterator, pr
            _, i_knowns = torch.max(prob, axis=1)
            i_knowns = i_knowns + 1
            probability_tensor = torch.zeros(prob.shape[0], 1001)
            probability_tensor[:, 1:] = prob
            P_max_all, _ = torch.max(prob, axis=1)
            pu = 1 - P_max_all
            probability_tensor[:, 0] = pu
            norm = torch.norm(probability_tensor, p=1, dim=1)
            normalized_tensor = probability_tensor / norm[:, None]
            p_max, i_max = torch.max(normalized_tensor, axis=1)
            index = (i_max == 0) * (i_knowns != y)
            y[index] = 1001
            L[k1:k2] = y

        is_bad = L == 1001
        data[:, 0] = L
        data_bad = data[is_bad]
        data_rest = data[~is_bad]

        torch.save(data_bad, f"/scratch/mjafarzadeh/feature_bad_b3_SP_{data_name}.pth")
        torch.save(data_rest, f"/scratch/mjafarzadeh/feature_rest_b3_SP_{data_name}.pth")

        del data, data_bad, data_rest, L, y, FV

        t5 = time.time()
        print(f"extracting {data_name} data set time = ", t5 - t4)


print("End")
