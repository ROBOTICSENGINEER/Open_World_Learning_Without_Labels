import torch
from collections import OrderedDict


class Centroid_Based_Concept_Learning(object):
    def __init__(self, top_k, threshold_distance, gpu=0):
        self.number_of_classes = 0
        self.total_number_of_centroid = 0
        self.all_classes_dict = OrderedDict()
        self.user_class_name_dict = OrderedDict()
        self.SoftMax = torch.nn.Softmax(dim=1).cuda(gpu)
        self.gpu = gpu
        self.threshold_distance = threshold_distance
        self.topk = top_k
        return

    def __len__(self):
        return self.number_of_classes

    def fit(self, class_to_process, features_dict):
        assert isinstance(features_dict, OrderedDict)

        for c in class_to_process:
            assert c not in self.user_class_name_dict
            self.number_of_classes = self.number_of_classes + 1
            class_dict_c = OrderedDict()
            f = features_dict[c]
            if f.dim() == 1:
                f = f.view(1, -1)
            if f.shape[0] == 1:
                class_dict_c["center"] = f.view(1, -1).detach().clone()
                class_dict_c["histogram_of_point"] = [1]
                class_dict_c["number_of_point"] = 1
                class_dict_c["number_of_center"] = 1
                self.total_number_of_centroid = self.total_number_of_centroid + 1
            else:
                center, hop = self.find_centers(f)
                class_dict_c["center"] = center.detach().clone()
                class_dict_c["histogram_of_point"] = hop
                class_dict_c["number_of_point"] = f.shape[0]
                class_dict_c["number_of_center"] = len(hop)
                self.total_number_of_centroid = self.total_number_of_centroid + len(hop)
            self.all_classes_dict[self.number_of_classes] = class_dict_c
            self.user_class_name_dict[c] = self.number_of_classes
        self.reduce_number_of_centroid()
        return

    def find_centers(self, FV):
        N = FV.shape[0]
        assert N > 1
        center_list = [FV[0, :]]
        histogram_of_point_list = [1]
        for n in range(1, N):
            f_n = FV[n : (n + 1), :]
            c_n = torch.vstack(center_list)
            d_n = torch.cdist(f_n, c_n, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            assert d_n.shape[0] == 1
            value_min, arg_min = torch.min(d_n, dim=1)
            if value_min <= self.threshold_distance:
                center_old = center_list[arg_min]
                wc = histogram_of_point_list[arg_min]
                center_new = ((wc * center_old) + f_n) / (wc + 1)
                center_list[arg_min] = center_new
                histogram_of_point_list[arg_min] = wc + 1
            else:
                center_list.append(f_n)
                histogram_of_point_list.append(1)
        if len(histogram_of_point_list) == 1:
            center = center_list[0].view(1, -1)
        else:
            center = torch.vstack(center_list)
        return (center, histogram_of_point_list)

    def reduce_number_of_centroid(self):
        # if number of centroid greater than threshold, 7500, reduce by kmeans
        # to be complete if needed
        return

    def update(self, class_to_process, features_dict):
        assert isinstance(features_dict, OrderedDict)
        for c in class_to_process:
            j = self.user_class_name_dict[c]
        raise NotImplementedError()
        return

    def predict(self, FV):
        if FV.dim() == 1:
            FV = FV.view(1, -1)

        if self.number_of_classes == 0:
            normalized_tensor = None

        elif self.number_of_classes == 1:
            normalized_tensor = torch.zeros(FV.shape[0], 2)
            # one class means that it is not trained
            normalized_tensor[:, 0] = 1.0

        elif self.number_of_classes == 2:
            normalized_tensor = torch.zeros(FV.shape[0], 3)
            # two class means that it is not trained
            normalized_tensor[:, 0] = 1.0

        else:
            distance_list = []
            class_id_list = []
            class_nc_list = []
            for k in range(self.number_of_classes):
                class_dict_k = self.all_classes_dict[k + 1]
                center_k = class_dict_k["center"].cuda(self.gpu)
                hop_k = class_dict_k["histogram_of_point"]
                np_k = class_dict_k["number_of_point"]
                nc_k = class_dict_k["number_of_center"]

                d_k = torch.cdist(FV, center_k, p=2, compute_mode="donot_use_mm_for_euclid_dist")
                distance_list.append(d_k)
                class_id_list = class_id_list + ([k + 1] * nc_k)
                class_nc_list = class_nc_list + ([np_k] * nc_k)
            d = torch.hstack(distance_list)

            top_k = min(self.topk, self.total_number_of_centroid)
            value_low_k, arg_low_k = torch.topk(d, top_k, dim=1, largest=False, sorted=True)
            is_distance_zero = torch.isclose(
                value_low_k[:, 0], torch.zeros(FV.shape[0], dtype=torch.double).cuda(self.gpu), atol=1e-04
            )

            scores = torch.ones(FV.shape[0], self.number_of_classes) * (-1000000)
            for i in range(FV.shape[0]):
                if is_distance_zero[i]:
                    j = arg_low_k[i, 0].item()
                    y = class_id_list[j]
                    scores[i, y - 1] = 10000.0
                else:
                    for k in range(top_k):
                        j = arg_low_k[i, k].item()
                        y = class_id_list[j]
                        d_ikj = d[i, j]
                        n_ikj = class_nc_list[j]
                        scores[i, y - 1] = max(0, scores[i, y - 1]) + (1 / (n_ikj * d_ikj))

            q = self.SoftMax(scores)
            # u = (value_low_k[:,0] - self.threshold_distance) / self.threshold_distance
            # u = torch.clamp(u, min = 0.0)
            m, _ = torch.max(q, dim=1)
            u = 1 - m
            probability_tensor = torch.zeros(FV.shape[0], 1 + self.number_of_classes)
            probability_tensor[:, 0] = u
            probability_tensor[:, 1:] = q
            norm = torch.norm(probability_tensor, p=1, dim=1)
            normalized_tensor = probability_tensor / norm[:, None]

        return normalized_tensor
