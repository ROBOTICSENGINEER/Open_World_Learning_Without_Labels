import torch
from collections import OrderedDict


class Nearest_Clasifier_Mean(object):
    def __init__(self, gpu=0):
        self.number_of_classes = 0
        self.center_dict = OrderedDict()
        self.user_class_name_dict = OrderedDict()
        self.SoftMax = torch.nn.Softmax(dim=1).cuda(gpu)
        self.gpu = gpu
        return

    def __len__(self):
        return self.number_of_classes

    def fit(self, class_to_process, features_dict):
        assert isinstance(features_dict, OrderedDict)
        for c in class_to_process:
            f = features_dict[c]
            if f.dim() == 1:
                center = f
            else:
                center = torch.mean(f, dim=0)
            self.number_of_classes = self.number_of_classes + 1
            self.center_dict[self.number_of_classes] = center
            self.user_class_name_dict[c] = self.number_of_classes

        if self.number_of_classes == 0:
            raise ValueError()
        elif self.number_of_classes == 1:
            self.centers = f.view(1, -1)
        else:
            class_id, centers = zip(*self.center_dict.items())
            self.centers = torch.vstack(list(centers))
        self.centers = self.centers.cuda(self.gpu)
        return

    def update(self, class_to_process, features_dict):
        assert isinstance(features_dict, OrderedDict)
        for c in class_to_process:
            j = self.user_class_name_dict[c]
            f = features_dict[c]
            if f.dim() == 1:
                center = f
            else:
                center = torch.mean(f, dim=0)
            self.center_dict[j] = center
        if self.number_of_classes == 1:
            self.centers = f.view(1, -1)
        else:
            class_id, centers = zip(*self.center_dict.items())
            self.centers = torch.vstack(list(centers))
        self.centers = self.centers.cuda(self.gpu)
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
            d = torch.cdist(FV, self.centers, p=2, compute_mode="donot_use_mm_for_euclid_dist")
            q = self.SoftMax(-d)
            m, _ = torch.max(q, dim=1)
            u = 1 - m
            probability_tensor = torch.zeros(FV.shape[0], 1 + self.number_of_classes)
            probability_tensor[:, 0] = u
            probability_tensor[:, 1:] = q
            norm = torch.norm(probability_tensor, p=1, dim=1)
            normalized_tensor = probability_tensor / norm[:, None]

        return normalized_tensor
