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

# main
if __name__ == '__main__':
    files_url = "https://ideami.com/llm_train"
    print(f"Downloading files from {files_url}")
    response = requests.get(files_url)
    zipfile.ZipFile(io.BytesIO(response.content)).extractall(".")