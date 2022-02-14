import time
import torch
from math import log2

t0 = time.time()

res = []
with torch.no_grad():
  for clustering in ["finch", "facto"]:
    for level in ["easy", "hard"]:
      for test_id in range(1,6):
        t1 = time.time()
        data = torch.load(f"/scratch/mjafarzadeh/feature_cluster_{clustering}_{level}_{test_id}.pth")
        for cluster_number, cluster in data.items():
          N_images = cluster['N_images']
          wnid_all = cluster['wnid_all']
          address = cluster['address']
          
          current_count = {w:0 for w in wnid_all}
          
          for A in address:
            w = A.split('/')[-2]
            current_count[w] +=1
          
          H = 0
          for c in current_count.values():
            if c > 0:
              p = c / N_images
              H = H - ( p * log2(p) )
            
          res.append((H, f"{clustering}_{level}_{test_id}_{cluster_number}"))


res.sort(key=lambda tup: tup[0]) 
for x, y in res:
  print(2**x , "   ", y)   
  
storage_dict = {}
for k, (h, y) in enumerate(res):
  clustering, level, test_id, cluster_number = y.split('_')
  data = torch.load(f"/scratch/mjafarzadeh/feature_cluster_{clustering}_{level}_{test_id}.pth")
  current_dict = data[int(cluster_number)]
  current_dict["clustering"] = clustering
  current_dict["level"] = level
  current_dict["test_id"] = test_id
  current_dict["cluster_id"] = int(cluster_number)
  current_dict["entropy"] = h
  current_dict["ranking"] = 2**h
  storage_dict[k+1] = current_dict
  
torch.save(storage_dict, "/scratch/mjafarzadeh/feature_clustering_ranked.pth")


t8 = time.time()
print("total time = ", t8 - t0)
