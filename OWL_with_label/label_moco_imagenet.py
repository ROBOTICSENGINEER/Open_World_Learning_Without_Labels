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

N_CPU = 100
batch_size = 100

name = "Label_mocoImagenet_tail33998_ct7_dm045_65"

tests_csv_root = "./preparing/all_test_166/"

efficientnet_b3_path = '/scratch/moco_imagenet_0199.pth.tar'
evm_model_path = '/scratch/EVM_cosine_model_imagnet_b3_mocoImagenet_tail33998_ct70.0_dm0.45.hdf5'

feature_size = 1536

tailsize = 33998
cover_threshold = 0.7
distance_multiplier = 0.45
unknown_dm = 0.65

N_known_classes = 1000
number_of_unknown_to_crate_evm = 5

csv_folder = './csv_folder/label_mocoImagenet_045_065'
cores = 32
detection_threshold = 0.001


levels = ['u10', 'u25', 'u50', 'u100']



class TextBasedDataset(torch.utils.data.Dataset):
  def __init__(self, text_file_path, data_root, transforms=None):
    with open(text_file_path) as f:
      self.samples = [line.rstrip() for line in f if line is not ''] 
    self.transforms = transforms
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

    if self.transforms is not None:
      image = self.transforms(img_pil)
    
    y = int(L)
    
    return image_path, image, y



class CIL(object):
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
    self.clustered_set= set() #empty set
    
    
    evm_known_feature_path = evm_model_path

    self.rho = number_of_unknown_to_crate_evm
    

    
    
    self.number_known_classes = N_known_classes 
    
    # All initialization happens here
    self.image_size = 300
    self.image_transform_test = transforms.Compose([
      transforms.Resize(size=(self.image_size,self.image_size), interpolation=Image.BICUBIC),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) ])
      
    # initialize feature extractor
    self.net = \
        net_model_from_lib(num_classes=N_known_classes)
    checkpoint = torch.load(efficientnet_b3_path, map_location="cpu")
    state_dict_pre = checkpoint['state_dict']
    
    # rename moco pre-trained keys
    
    print('keys = ', checkpoint.keys())
    state_dict = checkpoint['state_dict']
    for k in list(state_dict.keys()):
      # retain only encoder_q up to before the embedding layer
      if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
        # remove prefix
        state_dict[k[len("module.encoder_q."):]] = state_dict[k]
      # delete renamed or unused k
      del state_dict[k]
    
    msg = self.net.load_state_dict(state_dict, strict=False)
    assert set(msg.missing_keys) == {"classifier.weight", "classifier.bias"}
  
    for parameter in self.net.parameters():
      parameter.requires_grad = False
    self.net.to("cuda")
    self.net.eval()
 
    device = torch.device('cuda')
    self.net = torch.nn.DataParallel(self.net)
    self.net.to(device)
    
    # initialize EVM


    self.evm = MultipleEVM(tailsize=tailsize,
                 cover_threshold=cover_threshold,
                 distance_multiplier=distance_multiplier)
    self.evm.load(evm_model_path)
    


  def feature_extraction(self, dataset_path, dataset_root):
    # Create Dataloader for novelty detection
    
    num_workers = N_CPU
    dataset = TextBasedDataset(dataset_path, dataset_root, self.image_transform_test)
    loader = torch.utils.data.DataLoader(dataset, batch_size,  shuffle=False, num_workers=num_workers)
    self.features_dict = {}
    self.logit_dict = {}
    self.label_dict = {}
    for k, data in enumerate(loader):
      with autocast():
        image_paths, images, L = data
        image_features, image_Logits = self.net(images.to('cuda'))
        image_features = image_features.detach().cpu()
        image_Logits = image_Logits.detach().cpu()
        for image_name, image_feature, Logit, image_label in zip(image_paths, image_features, image_Logits, L):
          self.features_dict[image_name] = image_feature
          self.logit_dict[image_name] = Logit
          self.label_dict[image_name] = image_label.item()
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
    FVs = torch.stack(FVs)
    
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
    nu = 0 
    na = 0
    added_or_updated = set()
    
    if  self.red_light > 0.5:
      for im_name, label in self.label_dict.items():
        if label > N_known_classes:
          if label in self.residual_dict.keys():
            assert torch.is_tensor(self.features_dict[im_name])
            assert torch.is_tensor(self.residual_dict[label])
            self.residual_dict[label] =  torch.cat((self.residual_dict[label], self.features_dict[im_name].view(1,-1)), dim = 0)
            assert self.residual_dict[label].shape[1] == feature_size
            assert torch.is_tensor(self.residual_dict[label])
          else:
            self.residual_dict[label] = self.features_dict[im_name].view(1,-1)
            assert torch.is_tensor(self.residual_dict[label])
            assert self.residual_dict[label].shape[1] == feature_size
            
      
      if len(self.residual_dict)>= 1:
        
        for y, FV_positive in self.residual_dict.items():
          FV_negative_1 = torch.cat([self.residual_dict[k] for k in self.residual_dict.keys() if k != y], dim=0)
          
          FV_negative = FV_negative_1 #torch.from_numpy(FV_negative_1)
          if y in self.clustered_set:
            nu = nu + 1
            added_or_updated.add(y)
            #print("flag before update")
            if len(FV_negative_1)>0:
              self.evm.train_update(new_points = FV_positive, label = (y-1), distance_multiplier = unknown_dm , extra_negatives = FV_negative )
            else:
              self.evm.train_update(new_points = FV_positive, label = (y-1), distance_multiplier = unknown_dm )
          else:
            if FV_positive.shape[0] >= self.rho:
              na = na + 1
              added_or_updated.add(y)
              self.clustered_set.add(y)
              #print("flag before add")
              if len(FV_negative_1)>0:
                self.evm.train_update(new_points = FV_positive, label = (y-1), distance_multiplier = unknown_dm , extra_negatives = FV_negative )
              else:
                self.evm.train_update(new_points = FV_positive, label = (y-1), distance_multiplier = unknown_dm )
            #else:
            #  print(f"class {y} does not added/updated because FV_positive.shape[0] = {FV_positive.shape[0]} < {self.rho}")
      if (na + nu) > 0:
        for y in added_or_updated:
          del self.residual_dict[y]
      self.UU = self.UU + na
      print("End: len(self.clustered_set) = ", len(self.clustered_set))
      print("End: len(self.residual_dict) = ", len(self.residual_dict))
      print(f"{nu} evm classes updated.")
      print(f"{na} evm classes added.")
      print(f"Total discovered classes = {self.UU}")
    return
    
  def set_red_light(self, x):
      self.red_light = x
      return

  def save(self, level, test_id, name):
    self.evm.save(f'/scratch/CIL_{level}_{test_id}_EVM_{name}.hdf5')

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
    CIL_alg = CIL( test_id, csv_folder_i, cores, detection_threshold)
    
    t1 = time.time()
    print(f"Loading time {t1-t0}")
    start_time = time.time()
    
    num_rounds = (len(image_list)) //batch_size
    
    if ( (len(image_list)) % batch_size) !=0 :
      num_rounds += 1
    
    print("num_rounds = ", num_rounds)
    
    

    CIL_alg.set_red_light(0)
    for round_id in range(num_rounds):
      t2 = time.time()
      print(f"\nlevel {level}, test_id {test_id+1} from {number_of_tests}, red light off,  round_id {round_id+1} from {num_rounds}")
      round_file_name = name + "temp.csv"
      with open(round_file_name, "w") as f:
        f.writelines("\n".join(image_list[round_id*batch_size : (round_id+1)*batch_size]))
      t3 = time.time()
      CIL_alg.feature_extraction(dataset_path = round_file_name, dataset_root=None)
      t4 = time.time()
      CIL_alg.novelty_characterization(level, test_id, round_id)
      t5 = time.time()
      os.remove(round_file_name)
      t6 = time.time()
      print("feature_extraction time = ", t4-t3)
      print("novelty_characterization time = ", t5-t4)
      print("round time = ", t6-t2)

    

    CIL_alg.set_red_light(1)
    for round_id in range(num_rounds):
      t2 = time.time()
      print(f"\nlevel {level}, test_id {test_id+1} from {number_of_tests}, red light on,  round_id {round_id+1} from {num_rounds}")
      round_file_name = name + "temp.csv"
      with open(round_file_name, "w") as f:
        f.writelines("\n".join(image_list[round_id*batch_size : (round_id+1)*batch_size]))
      t3 = time.time()
      CIL_alg.feature_extraction(dataset_path = round_file_name, dataset_root=None)
      t4 = time.time()
      CIL_alg.novelty_characterization(level, test_id, round_id)
      t5 = time.time()
      CIL_alg.novelty_adaption(level, test_id, round_id)
      t6 = time.time()
      os.remove(round_file_name)
      t7 = time.time()
      print("feature_extraction time = ", t4-t3)
      print("novelty_characterization time = ", t5-t4)
      print("novelty_adaption time = ", t6-t5)
      print("round time = ", t7-t2)
      
    #CIL_alg.save(level = level, test_id = test_id, name = name)

    
    del CIL_alg
    end_time = time.time()
    print(f"Loading time {t1-t0}")
    print(f"run_{test_id} time {end_time-start_time}")
    print(f"Total time test_{test_id} {end_time-start_time+t1-t0}")



