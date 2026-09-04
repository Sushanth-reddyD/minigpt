import torch
import pytest
from minigpt.attention import MultiHeadAttention
from minigpt.config import GPT_CONFIG_124M


def make_small_cfg(**overrides):
    cfg = {
        "emb_dim": 16,
        "n_heads": 2,
        "context_length": 32,
        "qkv_bias": False,
        "drop_rate": 0.0,
    }
    cfg.update(overrides)
    return cfg


class TestMultiHeadAttention:
    def test_output_shape(self):
        mha = MultiHeadAttention(GPT_CONFIG_124M)
        x = torch.randn(2, 10, 768)
        out = mha(x)
        assert out.shape == (2, 10, 768)

    def test_output_shape_small(self):
        cfg = make_small_cfg()
        mha = MultiHeadAttention(cfg)
        x = torch.randn(1, 5, 16)
        out = mha(x)
        assert out.shape == (1, 5, 16)

    def test_no_qkv_bias(self):
        cfg = make_small_cfg(qkv_bias=False)
        mha = MultiHeadAttention(cfg)
        assert mha.W_q.bias is None
        assert mha.W_k.bias is None
        assert mha.W_v.bias is None

    def test_with_qkv_bias(self):
        cfg = make_small_cfg(qkv_bias=True)
        mha = MultiHeadAttention(cfg)
        assert mha.W_q.bias is not None
        assert mha.W_k.bias is not None
        assert mha.W_v.bias is not None

    def test_output_projection_has_bias(self):
        cfg = make_small_cfg()
        mha = MultiHeadAttention(cfg)
        assert mha.W_o.bias is not None

    def test_causal_mask_shape(self):
        cfg = make_small_cfg(context_length=64)
        mha = MultiHeadAttention(cfg)
        assert mha.mask.shape == (64, 64)

    def test_causal_mask_upper_triangular(self):
        cfg = make_small_cfg(context_length=4)
        mha = MultiHeadAttention(cfg)
        expected = torch.tensor([
            [False, True, True, True],
            [False, False, True, True],
            [False, False, False, True],
            [False, False, False, False],
        ])
        assert torch.equal(mha.mask, expected)

    def test_future_tokens_dont_affect_past(self):
        cfg = make_small_cfg()
        mha = MultiHeadAttention(cfg)
        mha.eval()

        x = torch.randn(1, 5, 16)
        out1 = mha(x)

        x_modified = x.clone()
        x_modified[0, 3, :] = torch.randn(16)
        x_modified[0, 4, :] = torch.randn(16)
        out2 = mha(x_modified)

        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6)

    def test_past_tokens_affect_future(self):
        cfg = make_small_cfg()
        mha = MultiHeadAttention(cfg)
        mha.eval()

        x = torch.randn(1, 5, 16)
        out1 = mha(x)

        x_modified = x.clone()
        x_modified[0, 0, :] = torch.randn(16)
        out2 = mha(x_modified)

        assert not torch.allclose(out1[0, 1], out2[0, 1], atol=1e-4)

    def test_first_token_attends_only_to_itself(self):
        cfg = make_small_cfg(n_heads=1)
        mha = MultiHeadAttention(cfg)
        mha.eval()

        x1 = torch.randn(1, 3, 16)
        out1 = mha(x1)

        x2 = x1.clone()
        x2[0, 0, :] = x1[0, 0, :]
        x2[0, 1, :] = torch.randn(16)
        x2[0, 2, :] = torch.randn(16)
        out2 = mha(x2)

        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-6)

    def test_batch_independence(self):
        cfg = make_small_cfg()
        mha = MultiHeadAttention(cfg)
        mha.eval()

        x = torch.randn(2, 5, 16)
        out = mha(x)

        x1 = x[0:1]
        out1 = mha(x1)

        assert torch.allclose(out[0], out1[0], atol=1e-6)

    def test_seq_shorter_than_context_length(self):
        cfg = make_small_cfg(context_length=64)
        mha = MultiHeadAttention(cfg)
        x = torch.randn(1, 5, 16)
        out = mha(x)
        assert out.shape == (1, 5, 16)

    def test_gradient_flows(self):
        cfg = make_small_cfg()
        mha = MultiHeadAttention(cfg)
        x = torch.randn(2, 5, 16, requires_grad=True)
        out = mha(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 5, 16)

    def test_dropout_active_in_train(self):
        cfg = make_small_cfg(drop_rate=0.99)
        mha = MultiHeadAttention(cfg)
        mha.train()
        x = torch.randn(1, 5, 16)
        out1 = mha(x)
        out2 = mha(x)
        assert not torch.equal(out1, out2)

    def test_dropout_inactive_in_eval(self):
        cfg = make_small_cfg(drop_rate=0.99)
        mha = MultiHeadAttention(cfg)
        mha.eval()
        x = torch.randn(1, 5, 16)
        out1 = mha(x)
        out2 = mha(x)
        assert torch.equal(out1, out2)

    def test_parameter_count_no_bias(self):
        cfg = make_small_cfg(emb_dim=64, n_heads=4, qkv_bias=False)
        mha = MultiHeadAttention(cfg)
        params = sum(p.numel() for p in mha.parameters())
        expected = 3 * (64 * 64) + (64 * 64 + 64)
        assert params == expected

    def test_d_k_calculation(self):
        cfg = make_small_cfg(emb_dim=768, n_heads=12)
        mha = MultiHeadAttention(cfg)
        assert mha.d_k == 64
        assert mha.n_heads == 12
