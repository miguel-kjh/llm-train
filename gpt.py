import torch
import torch.nn as nn

class GPT(nn.Module):
    def __init__(self, embed_size: int, context: int, vocab_size: int, n_layers: int, n_heads: int, BAIS: bool):
        super(GPT, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embed_size) # e.g. 4096, 384
        self.positions = nn.Embedding(context, embed_size) # e.g. 512, 384
        self.blocks = nn.Sequential(
            *[Block(embed_size, n_heads, BAIS) for _ in range(n_layers)]
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