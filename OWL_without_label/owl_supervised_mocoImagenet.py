import time
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
import os
from PIL import Image
from collections import OrderedDict
from ..finch import FINCH
import cv2

from ..MultipleEVM import MultipleEVM
from ..timm.models.efficientnet import efficientnet_b3 as net_model_from_lib
from torch.cuda.amp import autocast 


t0 = time.time()

number_of_tests = 5

N_CPU = 32
batch_size = 100

name = "supervised_mocoImagenet_tail33998_ct7_dm045_65"

tests_csv_root = "../data/all_test_166/"

csv_folder = './csv_folder/supervised_mocoImagenet_045_065'

efficientnet_b3_path_supervised = '/scratch/trained_efficientnet_b3_fp16_imagenet.pth.tar'
efficientnet_b3_path_mocoImagenet = '/scratch/moco_imagenet_0199.pth.tar'

evm_model_path = '/scratch/EVM_cosine_model_imagenet_b3_joint_supervised_mocoImagenet_tail33998_ct70_dm45.hdf5'

tailsize = 33998
cover_threshold = 0.7
distance_multiplier = 0.45
unknown_dm = 0.65

N_known_classes = 1000
number_of_unknown_to_strat_clustering = 50 
number_of_unknown_to_crate_evm = 5


cores = 32
detection_threshold = 0.001


levels = ['u10', 'u25', 'u50', 'u100']


class TextBasedDataset(torch.utils.data.Dataset):
  def __init__(self, text_file_path, data_root, transforms_supervised, transforms_moco):
    with open(text_file_path) as f:
      self.samples = [line.rstrip() for line in f if line is not ''] 
    self.transforms_supervised = transforms_supervised
    self.transforms_moco = transforms_moco
    self.data_root = data_root

  def __len__(self):
    return len(self.samples)

  def __getitem__(self, index):
    S = self.samples[index]
    image_path, L = S.split(',')
    if self.data_root is not None:
      img = cv2.imread(os.path.join(self.data_root, image_path), 1)
    else:
      img = cv2.imread(image_path, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img)

    image_supervised = self.transforms_supervised(img_pil)
    image_moco = self.transforms_moco(img_pil)
    
    y = int(L)
    
    return image_path, image_supervised, image_moco, y



class owl(object):
  def __init__(self, test_id,
         csv_folder, cores, detection_threshold):

    self.test_id = test_id
    
    self.red_light = 0


    self.csv_folder = csv_folder
    self.cores = cores
    self.detection_threshold = detection_threshold

    self.T = detection_threshold
    self.UU = 0
    
    self.residual_dict = {} #empty dictionary
    self.clustered_dict= {} #empty dictionary
    
    
    evm_known_feature_path = evm_model_path

    self.psi = number_of_unknown_to_strat_clustering
    self.rho = number_of_unknown_to_crate_evm
    

    
    
    self.number_known_classes = N_known_classes 
    
    # All initialization happens here
    self.image_size = 300
    
    self.image_transform_supervised = transforms.Compose([
      transforms.Resize(size=(self.image_size,self.image_size), interpolation=Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]) ])
      
    self.image_transform_moco = transforms.Compose([
      transforms.Resize(size=(self.image_size,self.image_size), interpolation=Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])

    # initialize feature extractor supervised
    self.net_supervised = \
        net_model_from_lib(num_classes=N_known_classes)
        
    # loading trained model supervised
    state_dict_model_supervised = torch.load(efficientnet_b3_path_supervised)
    new_state_dict_model_supervised = OrderedDict()
    for k, v in state_dict_model_supervised.items():
      name = k[7:] # remove `module.`
      new_state_dict_model_supervised[name] = v
    self.net_supervised.load_state_dict(new_state_dict_model_supervised)


    # initialize feature extractor moco Imagenet
    self.net_mocoImagenet = \
        net_model_from_lib(num_classes=N_known_classes)
    checkpoint_mocoImagenet = torch.load(efficientnet_b3_path_mocoImagenet, map_location="cpu")
    state_dict_pre_mocoImagenet = checkpoint_mocoImagenet['state_dict']
    
    # rename moco pre-trained keys moco Imagenet
    
    #print('keys = ', checkpoint.keys())
    state_dict_mocoImagenet = checkpoint_mocoImagenet['state_dict']
    for k in list(state_dict_mocoImagenet.keys()):
      # retain only encoder_q up to before the embedding layer
      if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        state_dict_mocoImagenet[k[len("module.encoder_q."):]] = state_dict_mocoImagenet[k]
      # delete renamed or unused k
      del state_dict_mocoImagenet[k]
    
    msg = self.net_mocoImagenet.load_state_dict(state_dict_mocoImagenet, strict=False)
    assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}


    for parameter in self.net_supervised.parameters():
      parameter.requires_grad = False
    for parameter in self.net_mocoImagenet.parameters():
      parameter.requires_grad = False


    self.net_supervised.to("cuda")
    self.net_mocoImagenet.to("cuda")
    self.net_supervised.eval()
    self.net_mocoImagenet.eval()

    device = torch.device('cuda')
    self.net_supervised = torch.nn.DataParallel(self.net_supervised)
    self.net_mocoImagenet = torch.nn.DataParallel(self.net_mocoImagenet)
    self.net_supervised.to(device)    
    self.net_mocoImagenet.to(device)

    
    # initialize EVM


    self.evm = MultipleEVM(tailsize=tailsize,
                 cover_threshold=cover_threshold,
                 distance_multiplier=distance_multiplier)
    self.evm.load(evm_model_path)
    


  def feature_extraction(self, dataset_path, dataset_root):
    # Create Dataloader for novelty detection
    
    num_workers = N_CPU
    dataset = TextBasedDataset(dataset_path, dataset_root, self.image_transform_supervised, self.image_transform_moco)
    loader = torch.utils.data.DataLoader(dataset, batch_size,  shuffle=False, num_workers=num_workers)
    self.features_dict = {}
    self.logit_dict = {}
    for k, data in enumerate(loader):
      with autocast():
        image_paths, images_supervised, images_moco, L = data
        FV_supervised, image_Logits_supervised = self.net_supervised(images_supervised.to('cuda'))
        FV_mocoImagenet, image_Logits_mocoImagene = self.net_mocoImagenet(images_moco.to('cuda'))

        image_features = np.hstack((FV_supervised.cpu().data.numpy(), FV_mocoImagenet.cpu().data.numpy()))
        
        for image_name, image_feature in zip(image_paths, image_features):
          self.features_dict[image_name] = image_feature
    return


  def novelty_characterization(self, level, test_id, round_id):

    # classify image to 1+K+UU classes
    
    if  self.red_light > 0.5:
      pre_post = "post"
    else:
      pre_post = "pre"
    
    result_path = os.path.join(self.csv_folder, 
                  f"characterization_{level}_{test_id}_{pre_post}_" + str(round_id).zfill(2)+".csv")
    image_names, FVs = zip(*self.features_dict.items())
    FVs = torch.from_numpy(np.array(FVs))
    
    Pr = self.evm.class_probabilities(FVs)
    Pr = torch.tensor(Pr)
    Pm,_ = torch.max(Pr, dim=1)
    pu = 1 - Pm
    all_rows_tensor = torch.cat((pu.view(-1,1), Pr), 1)
    norm = torch.norm(all_rows_tensor, p=1, dim=1)
    normalized_tensor = all_rows_tensor/norm[:,None]
    col1 = ['id', 'P_unknown']
    col2 = ['P_'+str(k) for k in range(1, self.number_known_classes+1)]
    col3 = ['U_'+str(k) for k in range(1, self.UU+1)]
    col = col1 + col2 + col3
    self.df_characterization = pd.DataFrame(zip(image_names,*normalized_tensor.t().tolist()), columns=col)
    self.df_characterization.to_csv(result_path, index = False, header = False, float_format='%.4f')
    
    result_path_raw = os.path.join(self.csv_folder, 
                  f"raw_characterization_{level}_{test_id}_{pre_post}_" + str(round_id).zfill(2)+".csv")
    self.df_characterization_raw = pd.DataFrame(zip(image_names,*all_rows_tensor.t().tolist()), columns=col)
    self.df_characterization_raw.to_csv(result_path_raw, index = False, header = False, float_format='%.4f')
    return result_path


  def novelty_adaption(self, level=None, test_id=None, round_id=None):
    """
    Update evm models
    :param features_dict (dict): A dictionary of image ids and image features
    :return none
    """
    m = -2
    nu = 0 
    if  self.red_light > 0.5:
      print("Start: len(self.clustered_dict) = ", len(self.clustered_dict))
      for k, row in self.df_characterization.iterrows():
        if row[1] > self.T:  # predicted unnkwon unknown
          self.residual_dict[row[0]] = self.features_dict[row[0]] 
      print("Start: len(self.residual_dict) = ", len(self.residual_dict))
      if len(self.residual_dict) >= self.psi:
        data =  np.vstack(self.residual_dict.values())
        c_all, num_clust, req_c = FINCH(data)
        cluster_labels = c_all[:,-1]
        m = num_clust[-1]  # number of clusters after clustering. 
        to_be_delete = []
        if m >= 2:
          image_names_residual, FVs_residual = zip(*self.residual_dict.items())
          if len(self.clustered_dict)>0:
            image_names_clustered, FVs_clustered = zip(*self.clustered_dict.items())
          else:
            FVs_clustered=[]
          for k in range(m):  # number of clusters after clustering. 
            index = [i for i in range(len(cluster_labels)) if cluster_labels[i] == k]
            index_neg = [i for i in range(len(cluster_labels)) if cluster_labels[i] != k]
            if len(index) > self.rho:
              to_be_delete = to_be_delete + index
              nu = nu+1
              FV_positive = torch.from_numpy(np.array([FVs_residual[k] for k in index]))
              FV_negative_1 = [FVs_residual[k] for k in index_neg]
              FV_negative_2 = list(FVs_clustered)
              FV_negative = torch.from_numpy(np.array(FV_negative_1 + FV_negative_2))
              y = self.number_known_classes + self.UU + nu
              # Train a new EVM with FV_positive and [FV_negative_1+FVs_clustered] as negatives
              # Insert the new EVM to new_EVM_list
              self.evm.train_update(new_points = FV_positive, label = (y-1), distance_multiplier = unknown_dm , extra_negatives = FV_negative )

            
        if nu > 0:
          image_covered = []
          for k in (to_be_delete):
            image_covered.append(image_names_residual[k])
          for name in image_covered:
            fv_name = self.residual_dict[name]
            self.clustered_dict.update({name:fv_name})
            del self.residual_dict[name]
        self.UU = self.UU + nu
      print("End: len(self.clustered_dict) = ", len(self.clustered_dict))
      print("End: len(self.residual_dict) = ", len(self.residual_dict))
      print(f"{nu} new evm classes added. Total discovered classes = {self.UU}")
    return
    
  def set_red_light(self, x):
      self.red_light = x
      return

  def save(self, level, test_id, name):
    self.evm.save(f'/scratch/owl_{level}_{test_id}_EVM_{name}.hdf5')

####################################

if not os.path.exists(csv_folder):
  os.makedirs(csv_folder)


for level in levels:
  for test_id in range(number_of_tests):
    t0 = time.time()
  
    with open(tests_csv_root + f'test_{level}_{test_id}.csv', "r") as f:
      image_list = f.readlines()
      image_list = list(map(str.strip, image_list))

    #csv_folder_i = csv_folder + f"/{level}_test_{test_id}"
    csv_folder_i = csv_folder
    owl_alg = owl( test_id, csv_folder_i, cores, detection_threshold)
    
    t1 = time.time()
    print(f"Loading time {t1-t0}")
    start_time = time.time()
    
    num_rounds = (len(image_list)) //batch_size
    
    if ( (len(image_list)) % batch_size) !=0 :
      num_rounds += 1
    
    print("num_rounds = ", num_rounds)
    
    

    owl_alg.set_red_light(0)
    for round_id in range(num_rounds):
      print(f"\n\level {level}, test_id {test_id+1} from {number_of_tests}, red light off,  round_id {round_id+1} from {num_rounds}")
      round_file_name = name + "temp.csv"
      with open(round_file_name, "w") as f:
        f.writelines("\n".join(image_list[round_id*batch_size : (round_id+1)*batch_size]))
      owl_alg.feature_extraction(dataset_path = round_file_name, dataset_root=None)
      owl_alg.novelty_characterization(level, test_id, round_id)
      os.remove(round_file_name)


    owl_alg.set_red_light(1)
    for round_id in range(num_rounds):
      print(f"\nlevel {level}, test_id {test_id+1} from {number_of_tests}, red light on,  round_id {round_id+1} from {num_rounds}")
      round_file_name = name + "temp.csv"
      with open(round_file_name, "w") as f:
        f.writelines("\n".join(image_list[round_id*batch_size : (round_id+1)*batch_size]))
      owl_alg.feature_extraction(dataset_path = round_file_name, dataset_root=None)
      owl_alg.novelty_characterization(level, test_id, round_id)
      owl_alg.novelty_adaption(level, test_id, round_id)
      os.remove(round_file_name)

    
    #owl_alg.save(level = level, test_id = test_id, name = name)

    
    del owl_alg
    end_time = time.time()
    print(f"Loading time {t1-t0}")
    print(f"run_{test_id} time {end_time-start_time}")
    print(f"Total time test_{test_id} {end_time-start_time+t1-t0}")



