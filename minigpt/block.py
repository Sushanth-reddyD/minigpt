import torch.nn as nn

from minigpt.layernorm import LayerNorm
from minigpt.attention import MultiHeadAttention
from minigpt.feedforward import FeedForward


class TransformerBlock(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.ln1 = LayerNorm(cfg["emb_dim"])
        self.attn = MultiHeadAttention(cfg)
        self.ln2 = LayerNorm(cfg["emb_dim"])
        self.ffn = FeedForward(cfg)
        self.drop_resid = nn.Dropout(cfg["drop_rate"])

    def forward(self, x):
        x = x + self.drop_resid(self.attn(self.ln1(x)))
        x = x + self.drop_resid(self.ffn(self.ln2(x)))
        return x
