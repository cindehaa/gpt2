from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn import functional as F



class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.ln_1(x)) # where they communicate
        x = x + self.mlp(self.ln_2(x)) # where they think
        return x

@dataclass
class GPTConfig:
    block_size: int = 256
    vocab_size: int = 65
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.transformer = nn.ModuleDict(dict{
            # token and position embeddings
            # nn.Embedding is just a wrapper around a block of numbers (like a single tensor)
            wte: nn.Embedding(config.vocab_size, config.n_embd), # output embedding
            wpe: nn.Embedding(config.block_size, config.n_embd), # positional encodings

            # transformer blocks (h probably stands for hidden)
            # list so we can index it
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # all the blocks

            # final layer norm
            ln_f: nn.LayerNorm(config.n_embd), # not in the diagram in the original paper; added by gpt2
        })

        # language model head, aka final classifier
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) # linear layer for language model head