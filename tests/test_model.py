import torch
import pytest
from minigpt.model import GPTModel
from minigpt.config import GPT_CONFIG_124M


def make_small_cfg(**overrides):
    cfg = {
        "vocab_size": 100,
        "context_length": 32,
        "emb_dim": 16,
        "n_heads": 2,
        "n_layers": 2,
        "drop_rate": 0.0,
        "qkv_bias": False,
    }
    cfg.update(overrides)
    return cfg


class TestGPTModel:
    def test_output_shape(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        ids = torch.randint(0, 100, (2, 10))
        logits = model(ids)
        assert logits.shape == (2, 10, 100)

    def test_output_shape_full_config(self):
        model = GPTModel(GPT_CONFIG_124M)
        ids = torch.randint(0, 50257, (1, 5))
        logits = model(ids)
        assert logits.shape == (1, 5, 50257)

    def test_logits_are_raw_scores(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        ids = torch.randint(0, 100, (1, 5))
        logits = model(ids)
        row = logits[0, 0]
        assert row.min() < 0 or row.max() > 1

    def test_weight_tying(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        assert model.lm_head.weight is model.tok_emb.weight

    def test_weight_tying_shapes(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        assert model.tok_emb.weight.shape == (100, 16)
        assert model.lm_head.weight.shape == (100, 16)

    def test_lm_head_no_bias(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        assert model.lm_head.bias is None

    def test_causal_future_doesnt_affect_past(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()

        ids = torch.randint(0, 100, (1, 5))
        logits1 = model(ids)

        ids_mod = ids.clone()
        ids_mod[0, 3] = (ids[0, 3] + 1) % 100
        ids_mod[0, 4] = (ids[0, 4] + 1) % 100
        logits2 = model(ids_mod)

        assert torch.allclose(logits1[0, :3], logits2[0, :3], atol=1e-5)

    def test_past_affects_future(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()

        ids = torch.randint(0, 100, (1, 5))
        logits1 = model(ids)

        ids_mod = ids.clone()
        ids_mod[0, 0] = (ids[0, 0] + 1) % 100
        logits2 = model(ids_mod)

        assert not torch.allclose(logits1[0, 2], logits2[0, 2], atol=1e-4)

    def test_batch_independence(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()

        ids = torch.randint(0, 100, (3, 5))
        logits = model(ids)

        logits0 = model(ids[0:1])
        logits1 = model(ids[1:2])

        assert torch.allclose(logits[0], logits0[0], atol=1e-5)
        assert torch.allclose(logits[1], logits1[0], atol=1e-5)

    def test_different_seq_lengths(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()

        ids3 = torch.randint(0, 100, (1, 3))
        ids10 = torch.randint(0, 100, (1, 10))

        logits3 = model(ids3)
        logits10 = model(ids10)

        assert logits3.shape == (1, 3, 100)
        assert logits10.shape == (1, 10, 100)

    def test_seq_up_to_context_length(self):
        cfg = make_small_cfg(context_length=16)
        model = GPTModel(cfg)
        ids = torch.randint(0, 100, (1, 16))
        logits = model(ids)
        assert logits.shape == (1, 16, 100)

    def test_gradient_flows(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        ids = torch.randint(0, 100, (1, 5))
        logits = model(ids)
        loss = logits.sum()
        loss.backward()
        for name, p in model.named_parameters():
            if p.requires_grad:
                assert p.grad is not None, f"no gradient for {name}"

    def test_parameter_count_124m(self):
        model = GPTModel(GPT_CONFIG_124M)
        total = sum(p.numel() for p in model.parameters())
        assert 100_000_000 < total < 200_000_000

    def test_n_blocks(self):
        cfg = make_small_cfg(n_layers=4)
        model = GPTModel(cfg)
        assert len(model.blocks) == 4

    def test_dropout_train_vs_eval(self):
        cfg = make_small_cfg(drop_rate=0.99)
        model = GPTModel(cfg)
        ids = torch.randint(0, 100, (1, 5))

        model.train()
        out1 = model(ids)
        out2 = model(ids)
        assert not torch.equal(out1, out2)

        model.eval()
        out3 = model(ids)
        out4 = model(ids)
        assert torch.equal(out3, out4)

    def test_position_embedding_used(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()

        ids = torch.tensor([[5, 5, 5, 5, 5]])
        logits = model(ids)
        assert not torch.allclose(logits[0, 0], logits[0, 1], atol=1e-4)
