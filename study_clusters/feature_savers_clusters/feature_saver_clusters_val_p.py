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
from timm.models.efficientnet import efficientnet_b3 as net_model_from_lib  # timm library
from math import log2


t0 = time.time()
_name_feature_extractor_ = "P"
folder = "/scratch/mjafarzadeh/result_val_p/"
output_folder = f"/scratch/mjafarzadeh/result_val_p/"
output_name = f"feature_cluster_val_p.pth"

batch_size = 512

number_of_known_classes = 1000
cnn_path_moco_places = "/scratch/mjafarzadeh/moco_places2_0199.pth"


n_cpu = 32

image_size = 300
N_feature = 1536 * 2
np.random.seed(2)
torch.manual_seed(2)


assert os.path.isfile(cnn_path_moco_places)


class list_data_class(Dataset):
    def __init__(self, image_list, transform_moco):
        self.samples = image_list
        self.transform_moco = transform_moco

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        A = self.samples[index]
        img = cv2.cvtColor(cv2.imread(A, 1), cv2.COLOR_BGR2RGB)
        img_pil = PIL.Image.fromarray(img)
        x_moco = self.transform_moco(img_pil)
        return x_moco


image_transform_moco = transforms.Compose(
    [
        transforms.Resize(size=(image_size, image_size), interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)


t1 = time.time()

model_moco_places = net_model_from_lib(num_classes=number_of_known_classes)  # timm library


assert os.path.isfile(cnn_path_moco_places)
checkpoint = torch.load(cnn_path_moco_places, map_location="cpu")
# rename moco pre-trained keys
print("keys = ", checkpoint.keys())
state_dict = checkpoint["state_dict"]
for k in list(state_dict.keys()):
    # retain only encoder_q up to before the embedding layer
    if k.startswith("module.encoder_q") and not k.startswith("module.encoder_q.fc"):
        # remove prefix
        state_dict[k[len("module.encoder_q.") :]] = state_dict[k]
    # delete renamed or unused k
    del state_dict[k]
msg = model_moco_places.load_state_dict(state_dict, strict=False)
assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
for parameter in model_moco_places.parameters():
    parameter.requires_grad = False
model_moco_places.cuda()
model_moco_places.eval()


t2 = time.time()
print("loading CNN time = ", t2 - t1)

torch.backends.cudnn.benchmark = True


with torch.no_grad():
    storage_dict = dict()
    entropy_list = []
    counter_storage = 0
    for u in [5, 10, 25, 50]:
        for test_id in range(1, 6):
            print(
                f"unknown {u} , test_id {test_id}",
            )

            file_list = []
            for file in os.listdir(folder):
                if f"track_val_p_u{u}_{test_id}_new_clusters_" in file:
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

            for cluster in range(1, 1 + N_cluster):
                # print(f"{cluster = }")
                current_dict = dict()
                images_list = cluster_image_address_dict[cluster]

                images_list_all = [A[18:] for A in images_list]
                images_list_known = [A for A in images_list_all if "ILSVRC_2012" in A]
                images_list_unknown = [A for A in images_list_all if not "ILSVRC_2012" in A]
                assert len(images_list) == len(images_list_known) + len(images_list_unknown)

                current_dict["extractor"] = _name_feature_extractor_
                current_dict["level"] = u
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

                current_dict["wnid_all"] = uni_all
                current_dict["wnid_known"] = uni_known
                current_dict["wnid_unknown"] = uni_unknown

                current_dict["N_images"] = len(images_list)
                current_dict["N_known_images"] = len(images_list_known)
                current_dict["N_unknown_images"] = len(images_list_unknown)
                current_dict["R_known_images"] = len(images_list_known) / len(images_list)
                current_dict["R_unknown_images"] = len(images_list_unknown) / len(images_list)
                current_dict["N_classes"] = len(uni_all)
                current_dict["N_known_classes"] = len(uni_known)
                current_dict["N_unknown_classes"] = len(uni_unknown)

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
                entropy_list.append((H, counter_storage))

                data_set = list_data_class(image_list=images_list, transform_moco=image_transform_moco)
                data_loader = DataLoader(dataset=data_set, batch_size=batch_size, shuffle=False, num_workers=n_cpu)
                N_data = data_set.__len__()

                for batch_id, x_moco in enumerate(data_loader, 0):
                    assert batch_id == 0
                    x_moco = x_moco.cuda()
                    FV_moco_places, _ = model_moco_places(x_moco)
                    FV = FV_moco_places
                    FV = FV.cpu()
                current_dict["feature"] = FV.detach().clone()
                storage_dict[counter_storage] = copy.deepcopy(current_dict)
                counter_storage = counter_storage + 1
    # print(storage_dict)

    entropy_list.sort(key=lambda tup: tup[0])
    sorted_storage_dict = dict()
    for k, (H, i) in enumerate(entropy_list):
        sorted_storage_dict[k + 1] = storage_dict[i]

    torch.save(sorted_storage_dict, output_folder + output_name)
