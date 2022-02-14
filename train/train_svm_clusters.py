import torch
import numpy as np
from sklearn.svm import LinearSVC
import pickle


for n_points in [1, 5]:
    for fe in ["s", "i", "p", "ip", "si", "sp", "sip"]:
        data_val = torch.load(f"/scratch/mjafarzadeh/result_val_{fe}/feature_cluster_val_{fe}.pth")
        # data is sorted base on entropy

        desired_thresh = 0.001 + data_val[n_points]["entropy"]
        print(f"feature = {fe}  ,  n_points = {n_points}  ,  threshold = {desired_thresh}")

        E_val = []
        VE_val = []
        VC_val = []
        for cluster_id, cluster in data_val.items():
            # if cluster['clustering'] != "finch": continue
            E_val.append(cluster["entropy"])
            FV = cluster["feature"]
            center = torch.mean(FV, dim=0)
            center = center.repeat(FV.shape[0], 1)
            de = FV - center
            ve = torch.linalg.matrix_norm(de, ord="fro") ** 2 / (FV.shape[0] - 1)
            VE_val.append(ve.item())

            F = cluster["feature"]
            norm_2 = torch.norm(F, p=2, dim=1)
            FV = F / norm_2[:, None]
            center = torch.mean(FV, dim=0)
            norm_2 = torch.linalg.vector_norm(center, ord=2)
            center = center / norm_2
            d = 1 - torch.mm(FV, center.view(-1, 1))
            vc = torch.linalg.matrix_norm(d, ord="fro") ** 2 / (FV.shape[0] - 1)
            VC_val.append(vc.item())

        E_val = np.array(E_val)
        VE_val = np.array(VE_val).reshape(-1, 1)
        VC_val = np.array(VC_val).reshape(-1, 1)
        Y_val = np.ones(E_val.shape)
        Y_val[E_val >= desired_thresh] = -1
        E_val = E_val.reshape(-1, 1)

        X_val = np.concatenate((VE_val, VC_val), axis=1)

        svm = LinearSVC(
            penalty="l2", class_weight="balanced", dual=False, random_state=0, tol=1e-5, max_iter=100000, C=100.0
        )
        svm.fit(X_val, Y_val)
        pickle.dump(svm, open(f"/scratch/mjafarzadeh/svm_val_{fe}_{n_points}point.pkl", "wb"))
        del svm
        del data_val
        del desired_thresh, X_val, Y_val, VE_val, VC_val, E_val
