import pytest
import torch

from minigpt.pretrained import load_gpt2, MODEL_CONFIGS


@pytest.fixture(scope="module")
def gpt2_model():
    return load_gpt2("gpt2")


def test_load_returns_gpt_model(gpt2_model):
    from minigpt.model import GPTModel

    assert isinstance(gpt2_model, GPTModel)


def test_parameter_count(gpt2_model):
    total = sum(p.numel() for p in gpt2_model.parameters())
    assert total == 124_439_808


def test_weight_tying_preserved(gpt2_model):
    assert gpt2_model.tok_emb.weight is gpt2_model.lm_head.weight


def test_embedding_not_zeros(gpt2_model):
    assert gpt2_model.tok_emb.weight.abs().sum().item() > 0


def test_position_embedding_not_zeros(gpt2_model):
    assert gpt2_model.pos_emb.weight.abs().sum().item() > 0


def test_qkv_bias_exists(gpt2_model):
    assert gpt2_model.blocks[0].attn.W_q.bias is not None
    assert gpt2_model.blocks[0].attn.W_k.bias is not None
    assert gpt2_model.blocks[0].attn.W_v.bias is not None


def test_forward_pass_shape(gpt2_model):
    input_ids = torch.tensor([[50256]])
    gpt2_model.eval()
    with torch.no_grad():
        logits = gpt2_model(input_ids)
    assert logits.shape == (1, 1, 50257)


def test_logits_match_huggingface(gpt2_model):
    from transformers import GPT2LMHeadModel

    input_ids = torch.tensor([[15496, 11, 616, 1438, 318]])

    gpt2_model.eval()
    with torch.no_grad():
        our_logits = gpt2_model(input_ids)

    hf_model = GPT2LMHeadModel.from_pretrained("gpt2", cache_dir="../data/hf_cache")
    hf_model.eval()
    with torch.no_grad():
        hf_logits = hf_model(input_ids).logits

    assert torch.allclose(our_logits, hf_logits, atol=1e-4)


def test_greedy_generation_coherent(gpt2_model):
    import tiktoken
    from minigpt.generate import generate

    tokenizer = tiktoken.get_encoding("gpt2")
    prompt = "Hello, my name is"
    input_ids = torch.tensor([tokenizer.encode(prompt)])

    gpt2_model.eval()
    with torch.no_grad():
        output_ids = generate(gpt2_model, input_ids, max_new_tokens=10, context_length=1024)

    text = tokenizer.decode(output_ids[0].tolist())
    assert text.startswith(prompt)
    assert len(text) > len(prompt)


def test_all_configs_defined():
    expected = {"gpt2", "gpt2-medium", "gpt2-large", "gpt2-xl"}
    assert set(MODEL_CONFIGS.keys()) == expected


def test_configs_have_required_keys():
    required = {"vocab_size", "context_length", "emb_dim", "n_heads", "n_layers", "drop_rate", "qkv_bias"}
    for name, cfg in MODEL_CONFIGS.items():
        assert required.issubset(cfg.keys()), f"{name} missing keys"
        assert cfg["qkv_bias"] is True
        assert cfg["drop_rate"] == 0.0
