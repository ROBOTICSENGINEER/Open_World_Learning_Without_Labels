import numpy as np
import pandas as pd
import os
import json
from copy import deepcopy

n_all = 5000
n_unknown = 2500
n_known = 2500
batch_size = 100
N_batch = 50

n_known_batch = 50

n_each_unknown_batch_dict = {5: 10, 10: 5, 25: 2, 50: 1}
n_each_unknown_total_dict = {5: 500, 10: 250, 25: 100, 50: 50}


df_known = pd.read_csv("imagenet_test_v2_93.csv", header=None, index_col=None)
df_unknown = pd.read_csv("imagenet_166.csv", header=None, index_col=None)

bad_image = "/scratch/datasets/ImageNet/ILSVRC_2010-360classes/train/n12765115/n12765115_14469.JPEG"
for i, x in enumerate(df_unknown.iloc[:, 0]):
    if x == bad_image:
        break
df_unknown = df_unknown.drop(i)

for u in [0, 5, 10, 25, 50]:
    if u > 0:
        df_index_classes = pd.read_csv(f"tests_v2/selected_test_u{u}.txt", header=None, index_col=None)
    else:
        df_index_classes = None

    for test_id in range(5):
        csv_file_output = f"test_v2_u{u}_{test_id+1}.csv"
        df_known = df_known.sample(frac=1, replace=False)
        if u == 0:
            d_test = df_known.iloc[:n_all, :]
        else:
            df_known = df_known.sample(frac=1, replace=False)
            d_known = deepcopy(df_known.iloc[:n_known, :])
            n_each_unknown_batch = n_each_unknown_batch_dict[u]
            n_each_unknown_total = n_each_unknown_total_dict[u]
            class_numbers_in_test = df_index_classes.iloc[test_id]
            u_dict = dict()
            for c in class_numbers_in_test:
                df_u_c = deepcopy(df_unknown[df_unknown.iloc[:, 1] == c])
                df_u_c = df_u_c.sample(n=n_each_unknown_total, replace=False)
                u_dict[c] = deepcopy(df_u_c)

            test_list = []
            for batch_id in range(N_batch):
                n1 = batch_id * n_known_batch
                n2 = (batch_id + 1) * n_known_batch
                m1 = batch_id * n_each_unknown_batch
                m2 = (batch_id + 1) * n_each_unknown_batch

                batch_list = [d_known.iloc[n1:n2:, :]]
                for c in class_numbers_in_test:
                    batch_list.append(u_dict[c].iloc[m1:m2, :])

                df_batch = pd.concat(batch_list, ignore_index=True)
                df_batch = df_batch.sample(frac=1, replace=False)
                test_list.append(df_batch)
            d_test = pd.concat(test_list, ignore_index=True)
        d_test.to_csv(csv_file_output, index=False, header=False)
