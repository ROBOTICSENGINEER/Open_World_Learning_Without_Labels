import torch
from collections import OrderedDict


class Gaussian_Mixture_Model(object):
    def __init__(self, scale=1.0):
        self.number_of_classes = 0
        self.center_dict = OrderedDict()
        self.covariance_dict = OrderedDict()
        self.inverse_covariance_dict = OrderedDict()
        self.user_class_name_dict = OrderedDict()
        self.SoftMax = torch.nn.Softmax(dim=1)
        self.scale = scale
        return

    def __len__(self):
        return self.number_of_classes

    def cov(self, x, m):
        # assume x diemention is n_sample x n_feature
        assert x.shape[0] > 1
        if m.dim() == 1:
            m = m.view(1, -1)
        y = x - m.expand(x.shape[0], -1)
        c = torch.mm(y.t(), y) / (x.shape[0] - 1)
        return c

    def inv_cov(self, covariance):
        det = torch.linalg.det(covariance)
        is_det_zero = torch.isclose(det, torch.zeros(1, dtype=torch.double), atol=1e-04)
        if is_det_zero:
            # u, s, ut = torch.linalg.svd(covariance)
            # N = max(1 , torch.sum(s>0.0001) )
            # u = u[:,0:N]
            # ut = ut[0:N,:]
            # covariance_prime = torch.mm( ut , torch.mm( covariance, u) )
            # covariance_prime_inverse = torch.linalg.inv(covariance_prime)
            # inv_covariance = torch.mm( u , torch.mm( covariance_prime_inverse, ut) )

            inv_covariance = torch.linalg.pinv(covariance)
            m = 1000 * torch.max(inv_covariance)
            if m == 0:
                inv_covariance = None
            else:
                for j in range(inv_covariance.shape[0]):
                    if torch.isclose(inv_covariance[j, j], torch.zeros(1, dtype=torch.double), atol=1e-05):
                        inv_covariance[j, j] = m
        else:
            inv_covariance = torch.linalg.inv(covariance)
        return inv_covariance

    def fit(self, class_to_process, features_dict):
        assert isinstance(features_dict, OrderedDict)
        for c in class_to_process:
            f = features_dict[c]
            if f.dim() == 1:
                center = None
                covariance = None
                inv_covariance = None
            elif f.shape[0] == 1:
                center = None
                covariance = None
                inv_covariance = None
            else:
                f = torch.unique(f, dim=0)
                if f.shape[0] == 1:
                    center = None
                    covariance = None
                    inv_covariance = None
                else:
                    center = torch.mean(f, dim=0)
                    covariance = self.cov(f, center)
                    inv_covariance = self.inv_cov(covariance)
                    if inv_covariance == None:
                        center = None
                        covariance = None
            self.number_of_classes = self.number_of_classes + 1
            self.center_dict[self.number_of_classes] = center
            self.covariance_dict[self.number_of_classes] = covariance
            self.inverse_covariance_dict[self.number_of_classes] = inv_covariance
            self.user_class_name_dict[c] = self.number_of_classes
        return

    def update(self, class_to_process, features_dict):
        assert isinstance(features_dict, OrderedDict)
        for c in class_to_process:
            j = self.user_class_name_dict[c]
            f = features_dict[c]
            if f.dim() == 1:
                center = None
                covariance = None
                inv_covariance = None
            elif f.shape[0] == 1:
                center = None
                covariance = None
                inv_covariance = None
            else:
                f = torch.unique(f, dim=0)
                if f.shape[0] == 1:
                    center = None
                    covariance = None
                    inv_covariance = None
                else:
                    center = torch.mean(f, dim=0)
                    covariance = self.cov(f, center)
                    inv_covariance = self.inv_cov(covariance)
                    if inv_covariance == None:
                        center = None
                        covariance = None
            self.center_dict[j] = center
            self.covariance_dict[j] = covariance
            self.inverse_covariance_dict[j] = inv_covariance
        return

    def predict(self, FV):
        if FV.dim() == 1:
            FV = FV.view(1, -1)

        if self.number_of_classes == 0:
            normalized_tensor = None

        else:
            q = torch.zeros(FV.shape[0], self.number_of_classes)
            for j, (k, mu) in enumerate(self.center_dict.items()):
                if mu != None:
                    inv_covariance_j = self.inverse_covariance_dict[k]
                    x_j = FV - mu.expand(FV.shape[0], -1)
                    for i in range(FV.shape[0]):
                        x_ij = x_j[i : (i + 1), :]
                        d_ij = torch.mm(x_ij, torch.mm(inv_covariance_j, x_ij.t()))
                        q[i, j] = torch.exp(-d_ij / (2 * self.scale))

            m, _ = torch.max(q, dim=1)
            u = 1 - m
            probability_tensor = torch.zeros(FV.shape[0], 1 + self.number_of_classes)
            probability_tensor[:, 0] = u
            probability_tensor[:, 1:] = q
            norm = torch.norm(probability_tensor, p=1, dim=1)
            normalized_tensor = probability_tensor / norm[:, None]

        return normalized_tensor
