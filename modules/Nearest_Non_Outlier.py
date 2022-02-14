import torch
import copy
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader
from scipy.special import gamma
import gc
import numpy


class OrderedDict_data_class(Dataset):
    def __init__(self, buffer_dict):
        self.label = []
        samples_list = []
        for c, FV in buffer_dict.items():
            L = c - 1
            self.label = self.label + ([L] * FV.shape[0])
            samples_list.append(FV)
        self.samples = torch.vstack(samples_list)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.samples[index, :]
        y = self.label[index]
        return (x, y)


class Nearest_classifier_Mean_Metric_learning(torch.nn.Module):
    def __init__(self, centers, rank=2, W=None):
        super(Nearest_classifier_Mean_Metric_learning, self).__init__()

        number_of_classes = centers.shape[0]
        feature_size = centers.shape[-1]
        self.C = centers.view(1, centers.shape[0], centers.shape[1]).detach().clone()
        self.C.requires_grad = False
        self.W = torch.nn.Linear(feature_size, rank, bias=False)
        if W != None:
            self.W.weight = torch.nn.Parameter(W.detach().clone())
            self.W.weight.requires_grad = True

    def forward(self, x):
        n = x.shape[0]
        m = self.C.shape[0]
        f = self.C.shape[-1]
        x = x.view(n, 1, f)
        _c_ = torch.repeat_interleave(self.C, n, dim=0)
        _x_ = torch.repeat_interleave(x, m, dim=1)
        y = _x_ - _c_
        h = self.W(y)
        d2 = torch.einsum("nmd,nmd->nm", h, h)
        Logit = -0.5 * d2
        return d2, Logit


class Nearest_Non_Outlier(object):
    def __init__(
        self,
        tau=1.0,
        rank=2,
        gpu=None,
        clipping_value=256,
        max_number_of_epoch=200,
        learning_rate=0.001,
        batch_size=8,
        num_workers=1,
    ):
        self.number_of_classes = 0
        self.center_dict = OrderedDict()
        self.user_class_name_dict = OrderedDict()
        self.rank = rank
        self.tau = tau
        self.gpu = gpu
        self.Buffer = OrderedDict()
        self.NCMML = None
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.learning_rate = learning_rate
        self.clipping_value = clipping_value
        self.max_number_of_epoch = max_number_of_epoch
        self.loss_old = 10.0**100
        self.loss_counter = 0
        self.recognition_weight = gamma((rank / 2) + 1) / ((numpy.pi ** (rank / 2)) * (tau**rank))
        self.old_w = None

        if gpu != None:
            self.SoftMax = torch.nn.Softmax(dim=1).cuda(gpu)
            self.loss_CE = torch.nn.CrossEntropyLoss(reduction="mean").cuda(gpu)
        else:
            self.SoftMax = torch.nn.Softmax(dim=1)
            self.loss_CE = torch.nn.CrossEntropyLoss(reduction="mean")
        return

    def __len__(self):
        return self.number_of_classes

    def fit(self, class_to_process, features_dict_new):
        assert isinstance(features_dict_new, OrderedDict)

        if len(features_dict_new) == 0:
            return
        N_old = self.number_of_classes
        N = self.number_of_classes + len(features_dict_new)
        if N < 3:
            return

        for c in class_to_process:
            f = features_dict_new[c]
            if f.dim() == 1:
                center = f.detach().clone()
                f = f.view(1, -1)
            else:
                center = torch.mean(f, dim=0)
            self.number_of_classes = self.number_of_classes + 1
            self.center_dict[self.number_of_classes] = center
            self.user_class_name_dict[c] = self.number_of_classes
            self.Buffer[self.number_of_classes] = f.detach().clone()

        if self.number_of_classes == 0:
            raise ValueError()
        elif self.number_of_classes == 1:
            self.centers = f.view(1, -1)
        else:
            class_id, centers = zip(*self.center_dict.items())
            self.centers = torch.vstack(list(centers))

        if self.gpu != None:
            self.centers = self.centers.cuda(self.gpu)

        dataset_train = OrderedDict_data_class(self.Buffer)

        train_loader = DataLoader(
            dataset=dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        if self.NCMML:
            self.old_w = self.NCMML.W.weight.detach().clone().cpu()
            del self.NCMML
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

        self.NCMML = Nearest_classifier_Mean_Metric_learning(centers=self.centers, rank=self.rank, W=self.old_w)

        self.NCMML.train()
        if self.gpu != None:
            self.NCMML.cuda(self.gpu)
        optimizer = torch.optim.Adam(self.NCMML.parameters(), lr=self.learning_rate)

        for epoch in range(self.max_number_of_epoch):
            for i, (x, y) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                if self.gpu != None:
                    x = x.cuda(self.gpu)
                    y = y.long().cuda(self.gpu)
                else:
                    y = y.long()
                d2, Logit = self.NCMML(x)
                loss = self.loss_CE(Logit, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.NCMML.parameters(), self.clipping_value)
                optimizer.step()
                if (self.loss_old - loss.item()) < 0.00001:
                    self.loss_counter = 1 + self.loss_counter
                else:
                    self.loss_counter = 0
                if self.loss_counter > 3:
                    break
                self.loss_old = copy.deepcopy(loss.detach().clone().cpu().item())

        return

    def update(self, class_to_process, features_dict_new):
        assert isinstance(features_dict, OrderedDict)
        raise NotImplimentedError()
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
            self.NCMML.eval()
            if self.gpu != None:
                FV = FV.cuda(self.gpu)
            else:
                FV = FV.cpu()
            with torch.no_grad():
                d2, Logit = self.NCMML(FV)
                softmax = self.SoftMax(Logit)
                d = torch.sqrt(d2)
                # print("inside NNO: d = ", d)
                f_hat = self.recognition_weight * (1 - (d / self.tau))
                # print("inside NNO: f_hat = ", f_hat)
                elementwise_known = (f_hat > 0).float()
                q = softmax * elementwise_known
                # print("inside NNO: q = ", q)
                m, _ = torch.max(q, dim=1)
                is_known = m > 0.0
                is_unknown = ~is_known

                if self.gpu != None:
                    probability_tensor = torch.zeros(FV.shape[0], 1 + self.number_of_classes).cuda(self.gpu)
                else:
                    probability_tensor = torch.zeros(FV.shape[0], 1 + self.number_of_classes)
                probability_tensor[is_unknown, 0] = 1.0
                probability_tensor[is_known, 1:] = q[is_known, :]
                norm = torch.norm(probability_tensor, p=1, dim=1)
                normalized_tensor = probability_tensor / norm[:, None]

        return normalized_tensor
