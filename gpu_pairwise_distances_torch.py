import torch
import time
import logging
import numpy as np
from pynvml import *
logger = logging.getLogger("GPU_Distances")

def get_no_of_splits(features, no_of_small_samples):
    # split chunks according to available GPU memory
    nvmlInit()
    h = nvmlDeviceGetHandleByIndex(0)
    info = nvmlDeviceGetMemoryInfo(h)
    gpu_free_mem = info.free
    size_in = features.nelement() * features.element_size()
    size_out = torch.zeros(features.shape[0],no_of_small_samples).nelement() * features.element_size()
    total_mem = ( size_in + size_out ) * 32
    if total_mem < gpu_free_mem: #no chunks if GPU mem is enough
        split = 1
    else:
        split = total_mem//gpu_free_mem + 3  # leave 20% GPU memory to avoid out of mem error
    return split

def cosine_distance(x1, x2=None, gpu=0, eps=1e-8):
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    if x2 is None:
        distances = 1 - torch.mm(x1, x1.t()) / (w1 * w1.t()).clamp(min=eps)
    else:
        x2 = x2.cuda(gpu)
        w2 = x2.norm(p=2, dim=1, keepdim=True)
        distances = 1 - torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)
    assert distances[distances<-eps].shape[0] == 0
    assert distances[distances>2+eps].shape[0] == 0
    return distances


def euclidean_distance(x1, x2=None, gpu=0, eps=1e-8):
    if x2 is None:
        distances = torch.cdist(x1, x1, p=2)
    else:
        x2 = x2.cuda(gpu)
        distances = torch.cdist(x1, x2, p=2)
    return distances



def compute_distance_chunk(smaller_tensor, bigger_tensor, splits, gpu, distance_function):
    t1 = time.time()
    
    if type(distance_function) == type(b"string"):
        distance_function_string =  distance_function.lower().decode("utf-8")
    elif type(distance_function) == type(np.bytes_('string')):
        distance_function_string =  distance_function.lower().decode("utf-8")
    else:
        distance_function_string = distance_function.lower()
    
    
    smaller_tensor = smaller_tensor.cuda(gpu)
    size_of_each_chunk = bigger_tensor.shape[0]//splits
    distances = torch.zeros(smaller_tensor.shape[0], bigger_tensor.shape[0])
    for indx in range(splits):
        logger.debug(f"Computing Distance: {indx+1} / {splits}")
        if distance_function_string == f'cosine':
            distances[:, indx*size_of_each_chunk:(indx+1)*size_of_each_chunk] = cosine_distance(smaller_tensor,
                                                                                            bigger_tensor[indx*size_of_each_chunk:(indx+1)*size_of_each_chunk,:])
        elif distance_function_string == f'euclidean':
            distances[:, indx*size_of_each_chunk:(indx+1)*size_of_each_chunk] = euclidean_distance(smaller_tensor,
                                                                                            bigger_tensor[indx*size_of_each_chunk:(indx+1)*size_of_each_chunk,:])
        else:
            print("You have entred distance_function = ", distance_function, " with type " , type(distance_function) 
                  , " as distance_function. However, distance_function can be either cosine or euclidean.")
            raise ValueError()
    if (splits*size_of_each_chunk)<bigger_tensor.shape[0]:
        if distance_function_string == 'cosine':
            distances[:, (indx+1)*size_of_each_chunk:] = cosine_distance(smaller_tensor,
                                   bigger_tensor[(indx+1)*size_of_each_chunk:,:])
        elif distance_function_string == 'euclidean':
            distances[:, (indx+1)*size_of_each_chunk:] = euclidean_distance(smaller_tensor,
                                   bigger_tensor[(indx+1)*size_of_each_chunk:,:])
        else:
            print("You have entred distance_function = ", distance_function, " with type " , type(distance_function) 
                  , " as distance_function. However, distance_function can be either cosine or euclidean.")
            raise ValueError() 
    t2 = time.time()
    logger.debug(f"GPU Distance Time {t2 - t1} seconds")
    return distances

def gpu_pairwise_distances(features1, features2=None, distance_function = 'cosine', gpu=0):
    if features1.shape[0] < features2.shape[0]:
        splits = get_no_of_splits(features2, features1.shape[0])
        result = compute_distance_chunk(features1.double(), features2.double(), splits, gpu, distance_function)
    else:
        splits = get_no_of_splits(features1, features2.shape[0])
        result = compute_distance_chunk(features2.double(), features1.double(), splits, gpu, distance_function)
        result = torch.transpose(result,1,0)
    del features1, features2
    return result

if __name__ == '__main__':
    logging.basicConfig(level=logging.DEBUG)
    gpu_pairwise_distances(features1 = torch.rand(10000,1000), features2 = torch.rand(1000000,1000), gpu = 0)
    # gpu_pairwise_distances(features1 = torch.rand(100,10), gpu = 0)
