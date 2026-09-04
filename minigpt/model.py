import torch
import torch.nn as nn

from minigpt.block import TransformerBlock
from minigpt.layernorm import LayerNorm


class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])
        self.blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )
        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.lm_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

        self.lm_head.weight = self.tok_emb.weight

    def forward(self, token_ids):
        batch, seq = token_ids.shape

        tok_emb = self.tok_emb(token_ids)
        pos_emb = self.pos_emb(torch.arange(seq, device=token_ids.device))
        x = self.drop_emb(tok_emb + pos_emb)

        x = self.blocks(x)
        x = self.final_norm(x)
        return self.lm_head(x)
