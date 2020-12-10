

#### unsupervised MoCo V2 training

The implementation of MoCo only supports **multi-gpu**, **DistributedDataParallel** training; single-gpu or DataParallel training is not supported.


***Note***: for training with 4 gpus, we recommend `--lr 0.015 --batch-size 128` . 

To do unsupervised MoCo V2 training of a EfficientNet-B3 model on ImageNet in an 4-gpu machine, run:
```
python main_moco_Imagenet.py  -lr 0.015  --batch-size 128 \
  --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 \
  --mlp --moco-t 0.2 --aug-plus --cos
```

