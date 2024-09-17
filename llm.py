import os, sys
from typing import Tuple
import ipdb
from tqdm import tqdm
from datetime import datetime
import platform, shutil
import requests, zipfile, io
from lightning import seed_everything

import torch
import torch.nn as nn
from torch.nn import functional as F

import sentencepiece as spm

from gpt import GPT

#deleted warnings
import warnings
warnings.filterwarnings("ignore")

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.cuda.empty_cache()

# Seed
SEED = 2024
seed_everything(SEED)

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


# loggin
wandb_log = False
wandb_project = 'llm_test'
wandb_run_name = 'llm1' + datetime.now().strftime("%Y%m%d-%H%M%S")

if wandb_log:
    import wandb
    wandb.init(project=wandb_project, name=wandb_run_name)

def load_data() -> str:
    folder = 'data'
    if not os.path.exists(folder):
        os.makedirs(folder)
    else:
        print(f"Data already exists in {folder}")
    with open(f'{folder}/wiki.txt', 'r', encoding='utf-8') as f:
        text = f.read()
    return text

def formating_num(num: int) -> str:
    return f"{num/1e6:.2f}"

def get_batch(data: torch.Tensor, batch_size: int, context: int) -> Tuple[torch.Tensor, torch.Tensor]:
    start = torch.randint(0, data.size(0) - context, (batch_size,))
    x = torch.stack([data[s:s+context] for s in start]) # (batch_size, context)
    y = torch.stack([data[s+1:s+1+context] for s in start]) # (batch_size, context)
    return x.to(device), y.to(device)

# main
if __name__ == '__main__':
    text = load_data()
    tokenizer_folder = 'tokenizer'
    sp = spm.SentencePieceProcessor(model_file=os.path.join(tokenizer_folder, 'wiki_tokenizer.model'))
    vocab_size = sp.get_piece_size()
    print("Vocab size: ", vocab_size)

    encode: callable = lambda x: sp.Encode(x)
    decode: callable = lambda x: sp.Decode(x)

    @torch.no_grad()
    def generate_samples(model: GPT, x: torch.Tensor, context: int, max: int = 500) -> torch.Tensor:
        x_ = torch.tensor(encode(x), dtype=torch.long, device=device)
        x_ = x_[None, :] # (1, context)
        new_samples = model.generate(x_, max, context)[0].tolist()
        return decode(new_samples)

    if os.path.exists("data/encoded_data.pt"):
        print("Loading encoded data")
        data = torch.load("data/encoded_data.pt")
    else:
        data = torch.tensor(encode(text), dtype=torch.long)
        torch.save(data, "data/encoded_data.pt")

    # data split
    data_size = len(data)
    spl = int(data_size * 0.9)
    train_data = data[:spl]
    val_data = data[spl:]

    print(f"Total data: {formating_num(data_size)} M | Train data: {formating_num(len(train_data))} M | Val data: {formating_num(len(val_data))} M")

    x, y = get_batch(train_data, batch_size, context)
    print(f"x: {x.shape} | y: {y.shape}")
    model = GPT(embed_size, context, vocab_size, n_layers, n_heads, BAIS, dropout).to(device)
    print(model)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {formating_num(parameters)} M")
    #logits, loss = model(x, y)
    #print(f"logits: {logits.shape} | loss: {loss.item()}")
    """idx = model.generate(x, 100, context)
    print(f"idx: {idx.shape}")"""
    input_ = "The quick brown fox jumps over the lazy dog"
    print(f"Input: {input_}")
    print(f"Generated: {generate_samples(model, input_, context, max=10)}")

    