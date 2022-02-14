import time
import numpy as np
import os
import PIL
from collections import OrderedDict
import cv2
import json
import copy
import gc
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from math import log2


t0 = time.time()
folder = "/scratch/mjafarzadeh/result_3_svm/"

with torch.no_grad():
    for level in ["easy", "hard"]:
        for test_id in range(1, 6):
            print(f"test {level} {test_id}")

            entropy_lit = []

            file_list = []
            for file in os.listdir(folder):
                if f"track_svm_hard_{test_id}_new_clusters_" in file:
                    file_list.append(os.path.join(folder, file))
            file_list.sort()

            cluster_image_address_dict = dict()
            N_cluster = 0

            for file in file_list:
                with open(file, "r") as f:
                    for L in f:
                        line = L.strip("\n")
                        if len(line) > 4:
                            if "cluster" in line:
                                cluster = int(line.split()[1])
                                cluster_image_address_dict[cluster] = list()
                                N_cluster = N_cluster + 1
                            elif "ImageNet" in line:
                                cluster_image_address_dict[cluster].append("/scratch/datasets/" + line)
                            else:
                                raise ValueError()

            for cluster in cluster_image_address_dict.keys():
                a = cluster_image_address_dict[cluster]
                cluster_image_address_dict[cluster] = sorted(a)
                # print("\n", sorted(a))

            for images_list in cluster_image_address_dict.values():
                current_dict = dict()

                images_list_all = [A[18:] for A in images_list]
                images_list_known = [A for A in images_list_all if "ILSVRC_2012" in A]
                images_list_unknown = [A for A in images_list_all if not "ILSVRC_2012" in A]
                assert len(images_list) == len(images_list_known) + len(images_list_unknown)

                current_dict["percentage"] = level
                current_dict["test_id"] = test_id
                current_dict["cluster_id"] = cluster
                current_dict["address"] = images_list_all

                uni_known = set()
                uni_unknown = set()
                for A in images_list_known:
                    wnid = A.split("/")[-2]
                    uni_known.add(wnid)
                for A in images_list_unknown:
                    wnid = A.split("/")[-2]
                    uni_unknown.add(wnid)

                uni_known = list(uni_known)
                uni_unknown = list(uni_unknown)
                uni_all = uni_known + uni_unknown

                N_images = len(images_list)
                current_count = {w: 0 for w in uni_all}
                for A in images_list_all:
                    w = A.split("/")[-2]
                    current_count[w] += 1
                H = 0
                for c in current_count.values():
                    if c > 0:
                        p = c / N_images
                        H = H - (p * log2(p))
                current_dict["entropy"] = H
                current_dict["ranking"] = 2**H
                entropy_lit.append(H)
            entropy_array = np.array(entropy_lit)
            print("min = ", np.min(entropy_array))
            print("max = ", np.max(entropy_array))
            print("median = ", np.median(entropy_array))
            print("mean = ", np.mean(entropy_array))
            print("std = ", np.std(entropy_array))
            print("sum = ", np.sum(entropy_array))
            print(" ")
