import torch
import pytest
from minigpt.layernorm import LayerNorm


class TestLayerNorm:
    def test_output_shape(self):
        ln = LayerNorm(emb_dim=768)
        x = torch.randn(2, 3, 768)
        out = ln(x)
        assert out.shape == (2, 3, 768)

    def test_normalized_mean_near_zero(self):
        ln = LayerNorm(emb_dim=64)
        x = torch.randn(4, 10, 64)
        out = ln(x)
        means = out.mean(dim=-1)
        assert torch.allclose(means, torch.zeros_like(means), atol=1e-5)

    def test_normalized_var_near_one(self):
        ln = LayerNorm(emb_dim=64)
        x = torch.randn(4, 10, 64)
        out = ln(x)
        var = out.var(dim=-1, correction=0)
        assert torch.allclose(var, torch.ones_like(var), atol=1e-4)

    def test_scale_init_ones(self):
        ln = LayerNorm(emb_dim=32)
        assert torch.equal(ln.scale, torch.ones(32))

    def test_shift_init_zeros(self):
        ln = LayerNorm(emb_dim=32)
        assert torch.equal(ln.shift, torch.zeros(32))

    def test_parameters_are_learnable(self):
        ln = LayerNorm(emb_dim=16)
        params = list(ln.parameters())
        assert len(params) == 2
        assert params[0].shape == (16,)  # scale
        assert params[1].shape == (16,)  # shift

    def test_each_token_normalized_independently(self):
        ln = LayerNorm(emb_dim=32)
        x = torch.randn(1, 2, 32)
        out = ln(x)
        x_modified = x.clone()
        x_modified[0, 1, :] = torch.randn(32)
        out_modified = ln(x_modified)
        assert torch.equal(out[0, 0], out_modified[0, 0])

    def test_learned_scale_shifts_output(self):
        ln = LayerNorm(emb_dim=4)
        ln.scale.data = torch.tensor([2.0, 2.0, 2.0, 2.0])
        ln.shift.data = torch.tensor([1.0, 1.0, 1.0, 1.0])
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])
        out = ln(x)
        var = out.var(dim=-1, correction=0)
        assert torch.allclose(var, torch.tensor([4.0]), atol=1e-4)
        mean = out.mean(dim=-1)
        assert torch.allclose(mean, torch.tensor([1.0]), atol=1e-4)

    def test_dry_run_example(self):
        ln = LayerNorm(emb_dim=4)
        x = torch.tensor([[[2.0, 4.0, 6.0, 8.0]]])
        out = ln(x)
        mean = 5.0
        var = 5.0
        expected = (torch.tensor([2.0, 4.0, 6.0, 8.0]) - mean) / (var + 1e-5) ** 0.5
        assert torch.allclose(out[0, 0], expected, atol=1e-5)

    def test_constant_input_returns_shift(self):
        ln = LayerNorm(emb_dim=4)
        x = torch.tensor([[[5.0, 5.0, 5.0, 5.0]]])
        out = ln(x)
        assert torch.allclose(out[0, 0], torch.zeros(4), atol=1e-4)

    def test_gradient_flows(self):
        ln = LayerNorm(emb_dim=16)
        x = torch.randn(2, 3, 16, requires_grad=True)
        out = ln(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 3, 16)
