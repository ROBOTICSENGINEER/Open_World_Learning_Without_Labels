import torch
import numpy as np
import matplotlib.pyplot as plt


data = torch.load("/scratch/mjafarzadeh/feature_clustering_ranked.pth")


# E = []
# V = []
# R = []
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   FV = cluster['feature']
#   center = torch.mean(FV, dim = 0)
#   center = center.repeat(FV.shape[0],1)
#   d = FV - center
#   v = torch.linalg.matrix_norm(d, ord = 'fro')**2 / (FV.shape[0]-1)
#   R.append(cluster['R_known_images']*100)
#   E.append(cluster['entropy'])
#   V.append(v)
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, E, marker = '.', c = R, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(R))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "% of knowns")
#
#
# ax.set_ylabel('Entropy')
# ax.set_xlabel('Variance from Center')
# plt.savefig(fname ='variance_center.png', dpi=300, bbox_inches='tight')
##############################################################
##############################################################
##############################################################
# E = []
# V = []
# R = []
#
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   F = cluster['feature']
#   norm_2 = torch.norm(F, p=2, dim=1)
#   FV = F/norm_2[:,None]
#   center = torch.mean(FV, dim = 0)
#   norm_2 = torch.linalg.vector_norm(center, ord=2)
#   center = center / norm_2
#   d = 1 - torch.mm(FV,center.view(-1,1))
#   v = torch.linalg.matrix_norm(d, ord = 'fro')**2 / (FV.shape[0]-1)
#   R.append(cluster['R_known_images']*100)
#   E.append(cluster['entropy'])
#   V.append(v)
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, E, marker = '.', c = R, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(R))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "% of knowns")
#
# ax.set_ylabel('Entropy')
# ax.set_xlabel('Variance from Center (cosine)')
# plt.savefig(fname ='variance_center_cosine.png', dpi=300, bbox_inches='tight')
##############################################################
##############################################################
##############################################################
# E = []
# V = []
# R = []
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   FV = cluster['feature']
#   P = torch.nn.functional.pdist(FV, p=2)
#   v = torch.sum(P*P) / ( ( FV.shape[0] -1 ) **2 )
#   R.append(cluster['R_known_images']*100)
#   E.append(cluster['entropy'])
#   V.append(v)
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, E, marker = '.', c = R, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(R))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "% of knowns")
#
# ax.set_ylabel('Entropy')
# ax.set_xlabel('Variance from each other')
# plt.savefig(fname ='variance_others.png', dpi=300, bbox_inches='tight')
##############################################################
##############################################################
##############################################################
# E = []
# V = []
# R = []
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   F = cluster['feature']
#   norm_2 = torch.norm(F, p=2, dim=1)
#   FV = F/norm_2[:,None]
#   P = 1 - torch.mm(FV,torch.transpose(FV, 0, 1))
#   v = torch.sum(P*P) / ( ( FV.shape[0] -1 ) **2 )
#   R.append(cluster['R_known_images']*100)
#   E.append(cluster['entropy'])
#   V.append(v)
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, E, marker = '.', c = R, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(R))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "% of knowns")
#
# ax.set_ylabel('Entropy')
# ax.set_xlabel('Variance from each other (cosine)')
# plt.savefig(fname ='variance_others_cosine.png', dpi=300, bbox_inches='tight')
##############################################################
##############################################################
##############################################################
# E = []
# V = []
# R = []
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   FV = cluster['feature']
#   _, S, _ = torch.svd(FV)
#   v = S[0] / torch.sum(S)
#   R.append(cluster['R_known_images']*100)
#   E.append(cluster['entropy'])
#   V.append(v)
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, E, marker = '.', c = R, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(R))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "% of knowns")
#
# ax.set_ylabel('Entropy')
# ax.set_xlabel('Max over sum singular value')
# plt.savefig(fname ='max_over_sum_singular_value.png', dpi=300, bbox_inches='tight')
##############################################################
##############################################################
##############################################################
# E = []
# V = []
# R = []
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   FV = cluster['feature']
#   _, S, _ = torch.svd(FV)
#   v = S[1] / S[0]
#   R.append(cluster['R_known_images']*100)
#   E.append(cluster['entropy'])
#   V.append(v.detach().clone().numpy())
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, E, marker = '.', c = R, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(R))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "% of knowns")
#
# ax.set_ylabel('Entropy')
# ax.set_xlabel('max over second singular value')
# plt.savefig(fname ='max_over_second_singular_value.png', dpi=300, bbox_inches='tight')
##############################################################
##############################################################
##############################################################
# E = []
# V = []
# C = []
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   FV = cluster['feature']
#   center = torch.mean(FV, dim = 0)
#   center = center.repeat(FV.shape[0],1)
#   d = FV - center
#   v = torch.linalg.matrix_norm(d, ord = 'fro')**2 / (FV.shape[0]-1)
#   _, S, _ = torch.svd(FV)
#   compactness = S[0] / torch.sum(S)
#   E.append(cluster['entropy'])
#   V.append(v.item())
#   C.append(compactness.item())
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, C, marker='.', c = E, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(E))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "Entropy")
# ax.set_ylabel('Max over sum singular value')
# ax.set_xlabel('Variance from Center')
# plt.savefig(fname ='variance_and_singular_value.png', dpi=300, bbox_inches='tight')
##############################################################
##############################################################
##############################################################
# E = []
# V = []
# C = []
# for cluster_id, cluster in data.items():
#   if cluster['clustering'] != "finch": continue
#   F = cluster['feature']
#   _, S, _ = torch.svd(F)
#   norm_2 = torch.norm(F, p=2, dim=1)
#   FV = F/norm_2[:,None]
#   center = torch.mean(FV, dim = 0)
#   norm_2 = torch.linalg.vector_norm(center, ord=2)
#   center = center / norm_2
#   d = 1 - torch.mm(FV,center.view(-1,1))
#   v = torch.linalg.matrix_norm(d, ord = 'fro')**2 / (FV.shape[0]-1)
#   compactness = S[0] / torch.sum(S)
#   E.append(cluster['entropy'])
#   V.append(v.item())
#   C.append(compactness.item())
#
# fig, ax = plt.subplots()
# im = ax.scatter(V, C, marker='.', c = E, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(E))
# cbar = fig.colorbar(im, ax=ax, shrink=0.95, label = "Entropy")
# ax.set_ylabel('Max over sum singular value')
# ax.set_xlabel('Variance from Center (cosine)')
# plt.savefig(fname ='variance_cosine_and_singular_value.png', dpi=300, bbox_inches='tight')

##############################################################
##############################################################
##############################################################
E = []
VE = []
VC = []
for cluster_id, cluster in data.items():
    if cluster["clustering"] != "finch":
        continue
    E.append(cluster["entropy"])
    FV = cluster["feature"]
    center = torch.mean(FV, dim=0)
    center = center.repeat(FV.shape[0], 1)
    de = FV - center
    ve = torch.linalg.matrix_norm(de, ord="fro") ** 2 / (FV.shape[0] - 1)

    F = cluster["feature"]
    norm_2 = torch.norm(F, p=2, dim=1)
    FV = F / norm_2[:, None]
    center = torch.mean(FV, dim=0)
    norm_2 = torch.linalg.vector_norm(center, ord=2)
    center = center / norm_2
    d = 1 - torch.mm(FV, center.view(-1, 1))
    vc = torch.linalg.matrix_norm(d, ord="fro") ** 2 / (FV.shape[0] - 1)

    VE.append(ve.item())
    VC.append(vc.item())

fig, ax = plt.subplots()
# im = ax.scatter(VE, VC, marker='.', c = E, cmap = "RdYlGn_r", vmin = 0.0, vmax = max(E))
im = ax.scatter(VE, VC, marker=".", c=E, cmap="RdYlGn_r", vmin=0.0, vmax=0.5)
cbar = fig.colorbar(im, ax=ax, shrink=0.95, label="Entropy")
ax.set_ylabel("Variance from Center Cosine")
ax.set_xlabel("Variance from Center Eucleadin")
plt.savefig(fname="variance_and_vaiance.png", dpi=300, bbox_inches="tight")
