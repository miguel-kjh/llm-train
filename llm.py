import os, sys
import ipdb
from tqdm import tqdm
from datetime import datetime
import platform, shutil
import requests, zipfile, io

import torch
import torch.nn as nn
from torch.nn import functional as F

import sentencepiece as spm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

# Hyperparameters architecture
batch_size = 8
context = 512
embed_size = 384
n_layers = 7
n_heads = 7
BAIS = True

# Hyperparameters
lr = 3e-4 # good starting point
dropout = 0.05
weight_decay = 0.01
grad_clip = 1.0

# Hyperparameters training
train_iters = 10e5
eval_interval = 50
eval_iters = 10
compile_ = False
checkpoint_dir = 'models'
checkpoint_fn = 'latest.pt'
checkpoint_load_fn = 'latest.pt'
dtype = torch.bfloat16

#Mode
inference = False

#device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")




# main
if __name__ == '__main__':
    pass