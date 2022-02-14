import time
import os
import numpy as np
import pandas as pd
import json
from B3 import calc_b3


folder_to_number_dict = json.load(open("./data/folder_to_id_dict_all.json"))


def get_label(x):

    f = x.split("/")[-2]
    if f in folder_to_number_dict:
        ind = int(folder_to_number_dict[f])
    else:
        raise ValueError(f"x is not in folder_to_number_dict")
    return ind


for U in [5, 10, 25, 50]:
    # for fe in ["s", "i", "p", "ip", "si", "sp", "sip"]:
    for fe in ["sp"]:

        score_list = []

        for test_id in range(1, 6):

            csv_path = f"/scratch/result_svm1_{fe}/track_svm1_{fe}_u{U}_{test_id}_output.csv"

            df = pd.read_csv(csv_path, header=None, index_col=False)
            address = df.iloc[:, 0]
            p = df.iloc[:, 1:].values
            y = np.argmax(p, axis=1)
            L = np.array([get_label(address[k]) for k in range(5000)])
            scores = np.zeros(41)

            for k in range(41):
                n1 = k * 100
                n2 = n1 + 1000

                predicted = y[n1:n2]
                Labels = L[n1:n2]

                is_known = Labels < 1001
                is_unknown = Labels >= 1001
                n_known = np.sum(is_known)
                n_unknown = np.sum(is_unknown)

                if n_known == 0:
                    f_measure, precision, recall = calc_b3(L=Labels, K=predicted)
                    scores[k] = f_measure
                elif n_unknown == 0:
                    acc = np.sum(Labels == predicted) / 1000
                    scores[k] = acc
                else:
                    predicted_known = (predicted < 1001) * (predicted > 0)
                    predicted_unknown = ~predicted_known
                    is_known_predicted_known = is_known * predicted_known
                    is_unknown_predicted_unknown = is_unknown * predicted_unknown
                    n_kk = np.sum(is_known_predicted_known)
                    n_uu = np.sum(is_unknown_predicted_unknown)
                    score_kk = 0.0
                    score_uu = 0.0
                    if n_kk > 0:
                        Labels_kk = Labels[is_known_predicted_known]
                        predicted_kk = predicted[is_known_predicted_known]
                        score_kk = np.sum(Labels_kk == predicted_kk)
                    if n_uu > 0:
                        Labels_uu = Labels[is_unknown_predicted_unknown]
                        predicted_uu = predicted[is_unknown_predicted_unknown]
                        f_measure, precision, recall = calc_b3(L=Labels_uu, K=predicted_uu)
                        score_uu = n_uu * f_measure
                    scores[k] = (score_kk + score_uu) / 1000
            score_list.append(scores[-1])
        print(f"{fe} {U} ,   {np.mean(np.array(score_list)):1.4f}  ,  {np.std(np.array(score_list)):1.4f}")

print("\nEnd\n")
