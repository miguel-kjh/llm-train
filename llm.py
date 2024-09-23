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
train_iters = 10e4
eval_interval = 50
eval_iters = 10
compile_ = False
load_pretrained = False
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
wandb_project = 'llm'
wandb_run_name = 'llm_normal' + datetime.now().strftime("%Y%m%d-%H%M%S")

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

def load_checkpoints(model: GPT, optimizer: torch.optim.Optimizer, checkpoint_dir: str, checkpoint_fn: str) -> Tuple[int, float]:
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    checkpoint_path = os.path.join(checkpoint_dir, checkpoint_fn)
    if os.path.exists(checkpoint_path) and load_pretrained:
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_iter = checkpoint['iter']
        best_val_loss = checkpoint['best_val_loss']
    else:
        print(f"No checkpoint found in {checkpoint_path}")
        start_iter = 0
        best_val_loss = float('inf')
    return start_iter, best_val_loss


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

    """x, y = get_batch(train_data, batch_size, context)
    print(f"x: {x.shape} | y: {y.shape}")
    model = GPT(embed_size, context, vocab_size, n_layers, n_heads, BAIS, dropout).to(device)
    print(model)
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {formating_num(parameters)} M")
    logits, loss = model(x, y)
    print(f"logits: {logits.shape} | loss: {loss.item()}")
    idx = model.generate(x, 100, context)
    print(f"idx: {idx.shape}")
    input_ = "The quick brown fox jumps over the lazy dog"
    print(f"Input: {input_}")
    print(f"Generated: {generate_samples(model, input_, context, max=10)}")"""

    # Training
    model = GPT(embed_size, context, vocab_size, n_layers, n_heads, BAIS, dropout)
    model.to(dtype)
    model.to(device)

    if compile_:
        print("Compiling model")
        model = torch.compile(model)
    
    parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {formating_num(parameters)} M")

    @torch.no_grad()
    def calculate_loss() -> dict:
        out = {}
        model.eval()
        for split, data in [("train", train_data), ("eval", val_data)]:
            l = torch.zeros(eval_iters)
            for i in range(eval_iters):
                x, y = get_batch(data, batch_size, context)
                _, loss = model(x, y)
                l[i] = loss
            out[split] = l.mean().item()
        model.train()
        return out
        
    p_dict = {p_name: p for p_name, p in model.named_parameters() if p.requires_grad}
    weight_decay_params = [p for n, p in p_dict.items() if p.dim() >= 2]
    no_decay_params = [p for n, p in p_dict.items() if p.dim() < 2]
    optimizer_groups = [
        {"params": weight_decay_params, "weight_decay": weight_decay},
        {"params": no_decay_params, "weight_decay": 0.0}
    ]

    optimizer = torch.optim.AdamW(optimizer_groups, lr=lr, betas=(0.9, 0.99))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, train_iters, eta_min=lr/10)
    
    start_iter, best_val_loss = load_checkpoints(model, optimizer, checkpoint_dir, checkpoint_load_fn)

    ## Inference
    if inference:
        model.eval()
        while True:
            input_ = input("Enter the beginning of a sentence: ")
            if input_ == "exit":
                break
            if len(input_) == 0:
                continue
            print("Generated: ", generate_samples(model, input_, context, max=100))
        sys.exit()

    # Training
    try:
        model.train()
        for i in tqdm(range(start_iter, int(train_iters)), initial=start_iter, total=int(train_iters)):
            x, y = get_batch(train_data, batch_size, context)
            logits, loss_batch = model(x, y)

            # Evaluation
            if (i % eval_interval == 0) or (i == train_iters - 1):
                loss = calculate_loss()
                print(f"Train loss: {loss['train']} | Eval loss: {loss['eval']}")
                print(generate_samples(model, "The quick brown fox jumps over the lazy dog", context, max=20))
                if loss['eval'] < best_val_loss:
                    print(f"Saving checkpoint to {checkpoint_dir}/{checkpoint_fn}")
                    best_val_loss = loss['eval']
                    torch.save({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'iter': i,
                        'best_val_loss': best_val_loss
                    }, os.path.join(checkpoint_dir, checkpoint_fn))

                if wandb_log:
                    wandb.log(
                        {
                            'train_loss': loss['train'], 
                            'eval_loss': loss['eval'], 
                            'lr': scheduler.get_last_lr()[0], 
                        },
                        step=i
                    )

            # Backward
            optimizer.zero_grad(set_to_none=True)
            loss_batch.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            scheduler.step()
        print("Training completed")

        if wandb_log:
            wandb.finish()
    except KeyboardInterrupt:
        print("Training interrupted")
        if wandb_log:
            wandb.finish()
        sys.exit(0)
    except Exception as e:
        print("Error: ", e)
    finally:
        torch.cuda.empty_cache()
        print("GPU memory cleaned")
        sys.exit(0)
    torch.cuda.empty_cache()




    