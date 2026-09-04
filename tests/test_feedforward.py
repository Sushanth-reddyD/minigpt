import torch
import pytest
from minigpt.feedforward import GELU, FeedForward
from minigpt.config import GPT_CONFIG_124M


class TestGELU:
    def test_positive_passes_through_nearly_unchanged(self):
        gelu = GELU()
        x = torch.tensor([3.0])
        out = gelu(x)
        assert out.item() == pytest.approx(3.0, abs=0.01)

    def test_negative_suppressed_not_killed(self):
        gelu = GELU()
        x = torch.tensor([-0.5])
        out = gelu(x)
        assert out.item() < 0
        assert out.item() > -0.5

    def test_large_negative_nearly_zero(self):
        gelu = GELU()
        x = torch.tensor([-3.0])
        out = gelu(x)
        assert abs(out.item()) < 0.01

    def test_zero_maps_to_zero(self):
        gelu = GELU()
        x = torch.tensor([0.0])
        out = gelu(x)
        assert out.item() == pytest.approx(0.0, abs=1e-7)

    def test_output_shape_preserved(self):
        gelu = GELU()
        x = torch.randn(2, 3, 768)
        out = gelu(x)
        assert out.shape == (2, 3, 768)

    def test_matches_pytorch_gelu(self):
        gelu = GELU()
        x = torch.linspace(-3, 3, 100)
        ours = gelu(x)
        theirs = torch.nn.functional.gelu(x, approximate="tanh")
        assert torch.allclose(ours, theirs, atol=1e-5)

    def test_gradient_flows_for_negative_input(self):
        gelu = GELU()
        x = torch.tensor([-1.0], requires_grad=True)
        out = gelu(x)
        out.backward()
        assert x.grad is not None
        assert x.grad.item() != 0.0

    def test_dry_run_positive(self):
        gelu = GELU()
        x = torch.tensor([2.0])
        out = gelu(x)
        assert out.item() == pytest.approx(1.9545, abs=0.001)

    def test_dry_run_negative(self):
        gelu = GELU()
        x = torch.tensor([-2.0])
        out = gelu(x)
        assert out.item() == pytest.approx(-0.0454, abs=0.001)


class TestFeedForward:
    def test_output_shape(self):
        ff = FeedForward(GPT_CONFIG_124M)
        x = torch.randn(2, 3, 768)
        out = ff(x)
        assert out.shape == (2, 3, 768)

    def test_inner_dimension_is_4x(self):
        ff = FeedForward(GPT_CONFIG_124M)
        first_linear = ff.layers[0]
        second_linear = ff.layers[2]
        assert first_linear.out_features == 4 * 768
        assert second_linear.in_features == 4 * 768

    def test_parameter_count(self):
        ff = FeedForward(GPT_CONFIG_124M)
        params = sum(p.numel() for p in ff.parameters())
        expected = (768 * 3072 + 3072) + (3072 * 768 + 768)
        assert params == expected

    def test_each_token_processed_independently(self):
        ff = FeedForward(GPT_CONFIG_124M)
        x = torch.randn(1, 2, 768)
        out = ff(x)
        x_modified = x.clone()
        x_modified[0, 1, :] = torch.randn(768)
        out_modified = ff(x_modified)
        assert torch.equal(out[0, 0], out_modified[0, 0])

    def test_gradient_flows(self):
        ff = FeedForward(GPT_CONFIG_124M)
        x = torch.randn(2, 3, 768, requires_grad=True)
        out = ff(x)
        loss = out.sum()
        loss.backward()
        assert x.grad is not None
        assert x.grad.shape == (2, 3, 768)

    def test_different_config(self):
        cfg = {"emb_dim": 256}
        ff = FeedForward(cfg)
        x = torch.randn(1, 5, 256)
        out = ff(x)
        assert out.shape == (1, 5, 256)
        assert ff.layers[0].out_features == 1024
