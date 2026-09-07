import torch
from transformers import GPT2LMHeadModel

from minigpt.model import GPTModel

MODEL_CONFIGS = {
    "gpt2": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 768,
        "n_heads": 12,
        "n_layers": 12,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
    "gpt2-medium": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1024,
        "n_heads": 16,
        "n_layers": 24,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
    "gpt2-large": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1280,
        "n_heads": 20,
        "n_layers": 36,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
    "gpt2-xl": {
        "vocab_size": 50257,
        "context_length": 1024,
        "emb_dim": 1600,
        "n_heads": 25,
        "n_layers": 48,
        "drop_rate": 0.0,
        "qkv_bias": True,
    },
}


def load_gpt2(model_size="gpt2"):
    cfg = MODEL_CONFIGS[model_size]
    model = GPTModel(cfg)

    hf_model = GPT2LMHeadModel.from_pretrained(
        model_size, cache_dir="../data/hf_cache"
    )
    hf = hf_model.state_dict()

    with torch.no_grad():
        model.tok_emb.weight.copy_(hf["transformer.wte.weight"])
        model.pos_emb.weight.copy_(hf["transformer.wpe.weight"])

        model.final_norm.scale.copy_(hf["transformer.ln_f.weight"])
        model.final_norm.shift.copy_(hf["transformer.ln_f.bias"])

        for i in range(cfg["n_layers"]):
            # ── QKV: fused Conv1D → three separate Linear ──
            qkv_w = hf[f"transformer.h.{i}.attn.c_attn.weight"].T
            q_w, k_w, v_w = qkv_w.chunk(3, dim=0)
            model.blocks[i].attn.W_q.weight.copy_(q_w)
            model.blocks[i].attn.W_k.weight.copy_(k_w)
            model.blocks[i].attn.W_v.weight.copy_(v_w)

            qkv_b = hf[f"transformer.h.{i}.attn.c_attn.bias"]
            q_b, k_b, v_b = qkv_b.chunk(3, dim=0)
            model.blocks[i].attn.W_q.bias.copy_(q_b)
            model.blocks[i].attn.W_k.bias.copy_(k_b)
            model.blocks[i].attn.W_v.bias.copy_(v_b)

            # ── Output projection: Conv1D → Linear ──
            model.blocks[i].attn.W_o.weight.copy_(
                hf[f"transformer.h.{i}.attn.c_proj.weight"].T
            )
            model.blocks[i].attn.W_o.bias.copy_(
                hf[f"transformer.h.{i}.attn.c_proj.bias"]
            )

            # ── Layer norms ──
            model.blocks[i].ln1.scale.copy_(
                hf[f"transformer.h.{i}.ln_1.weight"]
            )
            model.blocks[i].ln1.shift.copy_(
                hf[f"transformer.h.{i}.ln_1.bias"]
            )
            model.blocks[i].ln2.scale.copy_(
                hf[f"transformer.h.{i}.ln_2.weight"]
            )
            model.blocks[i].ln2.shift.copy_(
                hf[f"transformer.h.{i}.ln_2.bias"]
            )

            # ── FFN: Conv1D → Linear ──
            model.blocks[i].ffn.layers[0].weight.copy_(
                hf[f"transformer.h.{i}.mlp.c_fc.weight"].T
            )
            model.blocks[i].ffn.layers[0].bias.copy_(
                hf[f"transformer.h.{i}.mlp.c_fc.bias"]
            )
            model.blocks[i].ffn.layers[2].weight.copy_(
                hf[f"transformer.h.{i}.mlp.c_proj.weight"].T
            )
            model.blocks[i].ffn.layers[2].bias.copy_(
                hf[f"transformer.h.{i}.mlp.c_proj.bias"]
            )

    return model
