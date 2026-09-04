import torch
import torch.nn as nn


class MultiHeadAttention(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["emb_dim"]
        self.n_heads = cfg["n_heads"]
        self.d_k = d // self.n_heads

        self.W_q = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_k = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_v = nn.Linear(d, d, bias=cfg["qkv_bias"])
        self.W_o = nn.Linear(d, d)
        self.attn_dropout = nn.Dropout(cfg["drop_rate"])

        self.register_buffer(
            "mask",
            torch.triu(torch.ones(cfg["context_length"], cfg["context_length"]), diagonal=1).bool()
        )

    def forward(self, x):
        batch, seq, d = x.shape

        q = self.W_q(x)
        k = self.W_k(x)
        v = self.W_v(x)

        q = q.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(batch, seq, self.n_heads, self.d_k).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) / (self.d_k ** 0.5)
        scores = scores.masked_fill(self.mask[:seq, :seq], float("-inf"))
        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        context = weights @ v
        context = context.transpose(1, 2).contiguous().view(batch, seq, d)
        return self.W_o(context)
