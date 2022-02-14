import torch
import copy
from collections import OrderedDict
from torch.utils.data import Dataset, DataLoader


class OrderedDict_data_class(Dataset):
    def __init__(self, feature_dict, mapping_dict):
        self.label = []
        samples_list = []
        for c, FV in feature_dict.items():
            L = mapping_dict[c]
            self.label = self.label + ([L - 1] * FV.shape[0])
            samples_list.append(FV)
        self.samples = torch.vstack(samples_list)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, index):
        x = self.samples[index, :]
        y = self.label[index]
        return (x, y)


class LC(torch.nn.Module):
    def __init__(self, feature_size, num_classes):
        super(LC, self).__init__()
        self.fc = torch.nn.Linear(feature_size, num_classes)

    def forward(self, x):
        return self.fc(x)


class Scaling_Incremental_Learning(object):
    def __init__(
        self,
        feature_size,
        buffer_size=1000,
        softmax_threshold=0.5,
        learning_rate=0.001,
        batch_size=8,
        num_workers=1,
        max_number_of_epoch=200,
        gpu=None,
        clipping_value=256,
    ):
        self.number_of_classes = 0
        self.feature_size = feature_size
        self.user_class_name_dict = OrderedDict()
        self.gpu = gpu
        # self.classifer_initial_weight = OrderedDict()
        # self.classifer_initial_bias = OrderedDict()
        self.classifer_initial_mean_abs_bias = OrderedDict()
        self.classifer_initial_mean_sorted_abs_weight = OrderedDict()
        self.Buffer = OrderedDict()
        self.exampler_per_class = OrderedDict()
        self.max_number_of_epoch = max_number_of_epoch
        self.model = None
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.softmax_threshold = softmax_threshold
        self.buffer_size = buffer_size
        self.clipping_value = clipping_value
        if gpu != None:
            self.SoftMax = torch.nn.Softmax(dim=1).cuda(gpu)
        else:
            self.SoftMax = torch.nn.Softmax(dim=1)

        return

    def __len__(self):
        return self.number_of_classes

    def update_buffer(self, features_dict_new):
        assert isinstance(features_dict_new, OrderedDict)
        N_exampler = sum(list(self.exampler_per_class.values()))
        self.Buffer.update(features_dict_new)
        if N_exampler > self.buffer_size:
            desired_exampler_per_class = int(self.buffer_size / self.number_of_classes)
            if desired_exampler_per_class == 0:
                raise MemoryError("Error: buffer size of SCAIL is low. You should increase it at least by double")
            for k, v in self.Buffer.items():
                n = v.shape[0]
                if n > desired_exampler_per_class:
                    r = torch.randperm(n)[:desired_exampler_per_class]
                    self.Buffer[k] = v[r, :]
                    self.exampler_per_class[k] = desired_exampler_per_class
        return

    def fit(self, features_dict_new):
        assert isinstance(features_dict_new, OrderedDict)
        if len(features_dict_new) == 0:
            return
        N_old = self.number_of_classes
        N = self.number_of_classes + len(features_dict_new)
        if N < 3:
            return
        for c, f in features_dict_new.items():
            assert c not in self.user_class_name_dict
            if f.dim() == 1:
                features_dict_new[c] = f.view(1, -1)
            self.number_of_classes = self.number_of_classes + 1
            self.user_class_name_dict[c] = self.number_of_classes
            self.exampler_per_class[self.number_of_classes] = f.shape[0]

        current_features_dict = copy.deepcopy(self.Buffer)
        current_features_dict.update(features_dict_new)

        dataset_train = OrderedDict_data_class(current_features_dict, self.user_class_name_dict)

        train_loader = DataLoader(
            dataset=dataset_train, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers
        )

        new_classifier = LC(self.feature_size, N)

        new_classifier.train()
        if self.gpu != None:
            new_classifier.cuda(self.gpu)

        optimizer = torch.optim.Adam(new_classifier.parameters(), lr=self.learning_rate)

        if self.gpu != None:
            loss_CE = torch.nn.CrossEntropyLoss(reduction="sum").cuda(self.gpu)
        else:
            loss_CE = torch.nn.CrossEntropyLoss(reduction="sum")

        for epoch in range(self.max_number_of_epoch):
            for i, (x, y) in enumerate(train_loader, 0):
                optimizer.zero_grad()
                if self.gpu != None:
                    x = x.cuda(self.gpu)
                    y = y.long().cuda(self.gpu)
                else:
                    y = y.long()
                Logit = new_classifier(x)
                loss = loss_CE(Logit, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(new_classifier.parameters(), self.clipping_value)
                optimizer.step()

        with torch.no_grad():
            new_classifier.cpu()
            new_classifier.eval()
            bias = new_classifier.fc.bias
            weight = new_classifier.fc.weight

            abs_weight_new = torch.abs(weight[N_old:N, :])
            sorted_abs_weight_new, _ = torch.sort(abs_weight_new, dim=1, descending=True)
            mean_sorted_abs_weight_new = torch.mean(sorted_abs_weight_new, dim=0)
            mean_abs_bias_new = torch.mean(torch.abs(bias[N_old:N]))

            for i in range(N_old, N):
                # self.classifer_initial_weight[i] = copy.deepcopy(weight[i,:])
                # self.classifer_initial_bias[i] = copy.deepcopy(bias[i])
                self.classifer_initial_mean_abs_bias[i] = mean_abs_bias_new.item()
                self.classifer_initial_mean_sorted_abs_weight[i] = mean_sorted_abs_weight_new.detach().clone()
            self.model = copy.deepcopy(new_classifier)

            for i in range(N_old):
                #  Rectification
                initial_mean_abs_bias = self.classifer_initial_mean_abs_bias[i]
                bias_factor = mean_abs_bias_new / initial_mean_abs_bias
                self.model.fc.bias[i] = self.model.fc.bias[i] * bias_factor

                initial_mean_sorted_abs_weight = self.classifer_initial_mean_sorted_abs_weight[i]
                weights_factor = mean_sorted_abs_weight_new / initial_mean_sorted_abs_weight
                _, argsrt = torch.sort(weight[i, :], descending=True)
                for j in range(self.feature_size):
                    self.model.fc.weight[i, j] = self.model.fc.weight[i, j] * weights_factor[argsrt[j]]

            if self.gpu != None:
                self.model = self.model.cuda(self.gpu)
            self.update_buffer(features_dict_new)
            return

    def update(self, class_to_process, features_dict):
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
            if self.gpu != None:
                FV = FV.cuda(self.gpu)
            else:
                FV = FV.cpu()

            self.model.eval()
            with torch.no_grad():
                Logit = self.model(FV)
                softmax = self.SoftMax(Logit)
                m, _ = torch.max(softmax, dim=1)
                u = 1 - m
                probability_tensor = torch.zeros(FV.shape[0], 1 + self.number_of_classes)
                probability_tensor[:, 0] = u
                probability_tensor[:, 1:] = softmax
                norm = torch.norm(probability_tensor, p=1, dim=1)
                normalized_tensor = probability_tensor / norm[:, None]

        return normalized_tensor
