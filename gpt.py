import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    pass

class FeedForward(nn.Module):
    def __init__(self, embed_size: int, BAIS: bool, dropout: float, expansion: int = 6):
        super(FeedForward, self).__init__()
        inner_size = embed_size * expansion
        self.net = nn.Sequential(
            nn.Linear(embed_size, inner_size, bias=BAIS),
            nn.GELU(),
            nn.Linear(inner_size, embed_size, bias=BAIS),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x) 

class Block(nn.Module):

    def __init__(self, embed_size: int, n_heads: int, BAIS: bool, dropout: float):
        super(Block, self).__init__()
        head_size = embed_size // n_heads
        self.attn = MultiHeadAttention(n_heads, head_size)
        self.ffn = FeedForward(embed_size, BAIS, dropout)
        self.ln1 = nn.LayerNorm(embed_size)
        self.ln2 = nn.LayerNorm(embed_size)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, embed_size: int, context: int, vocab_size: int, n_layers: int, n_heads: int, BAIS: bool, dropout: float):
        super(GPT, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size) # e.g. 4096, 384
        self.positions = nn.Embedding(context, embed_size) # e.g. 512, 384
        self.blocks = nn.Sequential(
            *[Block(embed_size, n_heads, BAIS, dropout) for _ in range(n_layers)]
        )
        self.ln = nn.LayerNorm(embed_size)
        self.final_layer = nn.Linear(embed_size, vocab_size, bias=BAIS) # e.g. 384, 4096
        self.apply(self._init_weights)

    #parameters initialization
    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, x, targets = None):
        # x: (batch_size, context)
        loss = None
        BS, SL = x.shape
        emb = self.embeddings(x) # (batch_size, context, embed_size)
        pos = self.positions(torch.arange(SL).to(x.device)) # (context, embed_size)
        x = emb + pos
        x = self.blocks(x) # (batch_size, context, embed_size)
        x = self.ln(x) # (batch_size, context, embed_size)
        self.logits = self.final_layer(x) # (batch_size, context, vocab_size)
        if targets is not None:
            # targets: (batch_size, context)
            # self.logits: (batch_size, context, vocab_size) -> (batch_size*context, vocab_size)
            loss = nn.CrossEntropyLoss()(self.logits.view(-1, self.logits.size(-1)), targets.view(-1))
        return self.logits, loss
    
    # generate new samples
    def generate(self, x, max: int, context: int):
        for _ in range(max):
            x = x[:, -context:] # (1, context)
            logits, _ = self(x)
            logits = logits[:, -1, :]
            probs = nn.functional.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            x = torch.cat((x, next_token), dim=1)
        return x