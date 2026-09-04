import torch
import pytest
from minigpt.block import TransformerBlock
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


class TestTransformerBlock:
    def test_output_shape(self):
        block = TransformerBlock(GPT_CONFIG_124M)
        x = torch.randn(2, 10, 768)
        out = block(x)
        assert out.shape == (2, 10, 768)

    def test_output_shape_small(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        x = torch.randn(1, 5, 16)
        out = block(x)
        assert out.shape == (1, 5, 16)

    def test_residual_preserves_input_signal(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        block.eval()
        x = torch.randn(1, 3, 16)
        out = block(x)
        diff = (out - x).abs().mean()
        assert diff < 10.0

    def test_not_identity(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        block.eval()
        x = torch.randn(1, 3, 16)
        out = block(x)
        assert not torch.allclose(out, x, atol=1e-5)

    def test_causal_future_doesnt_affect_past(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        block.eval()

        x = torch.randn(1, 5, 16)
        out1 = block(x)

        x_mod = x.clone()
        x_mod[0, 3, :] = torch.randn(16)
        x_mod[0, 4, :] = torch.randn(16)
        out2 = block(x_mod)

        assert torch.allclose(out1[0, :3], out2[0, :3], atol=1e-6)

    def test_past_affects_future(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        block.eval()

        x = torch.randn(1, 5, 16)
        out1 = block(x)

        x_mod = x.clone()
        x_mod[0, 0, :] = torch.randn(16)
        out2 = block(x_mod)

        assert not torch.allclose(out1[0, 2], out2[0, 2], atol=1e-4)

    def test_batch_independence(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        block.eval()

        x = torch.randn(3, 5, 16)
        out = block(x)

        out0 = block(x[0:1])
        out1 = block(x[1:2])

        assert torch.allclose(out[0], out0[0], atol=1e-6)
        assert torch.allclose(out[1], out1[0], atol=1e-6)

    def test_has_two_layernorms(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        assert hasattr(block, "ln1")
        assert hasattr(block, "ln2")
        assert block.ln1.scale.shape == (16,)
        assert block.ln2.scale.shape == (16,)

    def test_stacking_blocks(self):
        cfg = make_small_cfg()
        blocks = [TransformerBlock(cfg) for _ in range(3)]
        x = torch.randn(1, 5, 16)
        for block in blocks:
            x = block(x)
        assert x.shape == (1, 5, 16)

    def test_gradient_flows_through_residual(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        x = torch.randn(1, 5, 16, requires_grad=True)
        out = block(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert (x.grad.abs() > 0).all()

    def test_dropout_train_vs_eval(self):
        cfg = make_small_cfg(emb_dim=64, n_heads=4, drop_rate=0.5)
        block = TransformerBlock(cfg)

        x = torch.randn(4, 20, 64)

        block.train()
        out_train1 = block(x)
        out_train2 = block(x)
        assert not torch.equal(out_train1, out_train2)

        block.eval()
        out_eval1 = block(x)
        out_eval2 = block(x)
        assert torch.equal(out_eval1, out_eval2)

    def test_pre_norm_ordering(self):
        cfg = make_small_cfg()
        block = TransformerBlock(cfg)
        block.eval()
        x = torch.randn(1, 3, 16)
        normed = block.ln1(x)
        attn_out = block.attn(normed)
        expected_after_attn = x + attn_out
        normed2 = block.ln2(expected_after_attn)
        ffn_out = block.ffn(normed2)
        expected = expected_after_attn + ffn_out
        actual = block(x)
        assert torch.allclose(actual, expected, atol=1e-6)
