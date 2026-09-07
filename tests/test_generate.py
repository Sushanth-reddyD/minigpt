import torch
import pytest
from minigpt.generate import generate
from minigpt.model import GPTModel


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


class TestGenerate:
    def test_output_length(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        out = generate(model, prompt, max_new_tokens=5, context_length=32)
        assert out.shape == (1, 8)

    def test_prompt_preserved(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[10, 20, 30]])
        out = generate(model, prompt, max_new_tokens=4, context_length=32)
        assert torch.equal(out[0, :3], prompt[0])

    def test_generates_valid_token_ids(self):
        cfg = make_small_cfg(vocab_size=50)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2]])
        out = generate(model, prompt, max_new_tokens=10, context_length=32)
        generated = out[0, 2:]
        assert (generated >= 0).all()
        assert (generated < 50).all()

    def test_deterministic(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[5, 10, 15]])
        out1 = generate(model, prompt.clone(), max_new_tokens=8, context_length=32)
        out2 = generate(model, prompt.clone(), max_new_tokens=8, context_length=32)
        assert torch.equal(out1, out2)

    def test_different_prompts_different_outputs(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt1 = torch.tensor([[1, 2, 3]])
        prompt2 = torch.tensor([[50, 60, 70]])
        out1 = generate(model, prompt1, max_new_tokens=5, context_length=32)
        out2 = generate(model, prompt2, max_new_tokens=5, context_length=32)
        assert not torch.equal(out1[0, 3:], out2[0, 3:])

    def test_zero_new_tokens(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        out = generate(model, prompt, max_new_tokens=0, context_length=32)
        assert torch.equal(out, prompt)

    def test_context_window_truncation(self):
        cfg = make_small_cfg(context_length=4)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        out = generate(model, prompt, max_new_tokens=5, context_length=4)
        assert out.shape == (1, 8)

    def test_long_generation_beyond_context(self):
        cfg = make_small_cfg(context_length=8)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2]])
        out = generate(model, prompt, max_new_tokens=15, context_length=8)
        assert out.shape == (1, 17)

    def test_batch_generation(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3], [4, 5, 6]])
        out = generate(model, prompt, max_new_tokens=4, context_length=32)
        assert out.shape == (2, 7)

    def test_greedy_picks_argmax(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])

        logits = model(prompt)
        expected_next = torch.argmax(logits[0, -1, :]).item()

        out = generate(model, prompt.clone(), max_new_tokens=1, context_length=32)
        actual_next = out[0, 3].item()
        assert actual_next == expected_next

    def test_single_token_prompt(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[42]])
        out = generate(model, prompt, max_new_tokens=5, context_length=32)
        assert out.shape == (1, 6)
        assert out[0, 0].item() == 42


class TestTemperature:
    def test_temperature_zero_is_greedy(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        greedy = generate(model, prompt.clone(), max_new_tokens=5, context_length=32)
        temp0 = generate(model, prompt.clone(), max_new_tokens=5, context_length=32, temperature=0.0)
        assert torch.equal(greedy, temp0)

    def test_temperature_produces_valid_tokens(self):
        cfg = make_small_cfg(vocab_size=50)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2]])
        out = generate(model, prompt, max_new_tokens=10, context_length=32, temperature=1.0)
        generated = out[0, 2:]
        assert (generated >= 0).all()
        assert (generated < 50).all()

    def test_temperature_output_length(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        out = generate(model, prompt, max_new_tokens=5, context_length=32, temperature=0.8)
        assert out.shape == (1, 8)

    def test_low_temperature_less_random(self):
        torch.manual_seed(42)
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])

        results_low = set()
        results_high = set()
        for _ in range(20):
            out_low = generate(model, prompt.clone(), max_new_tokens=1, context_length=32, temperature=0.01)
            out_high = generate(model, prompt.clone(), max_new_tokens=1, context_length=32, temperature=5.0)
            results_low.add(out_low[0, -1].item())
            results_high.add(out_high[0, -1].item())

        assert len(results_low) <= len(results_high)


class TestTopK:
    def test_top_k_valid_tokens(self):
        cfg = make_small_cfg(vocab_size=50)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2]])
        out = generate(model, prompt, max_new_tokens=10, context_length=32, temperature=1.0, top_k=5)
        generated = out[0, 2:]
        assert (generated >= 0).all()
        assert (generated < 50).all()

    def test_top_k_1_is_greedy(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        greedy = generate(model, prompt.clone(), max_new_tokens=5, context_length=32)
        topk1 = generate(model, prompt.clone(), max_new_tokens=5, context_length=32, temperature=1.0, top_k=1)
        assert torch.equal(greedy, topk1)

    def test_top_k_limits_choices(self):
        torch.manual_seed(42)
        cfg = make_small_cfg(vocab_size=20)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2]])

        tokens_seen = set()
        for _ in range(50):
            out = generate(model, prompt.clone(), max_new_tokens=1, context_length=32, temperature=1.0, top_k=3)
            tokens_seen.add(out[0, -1].item())

        assert len(tokens_seen) <= 3


class TestTopP:
    def test_top_p_valid_tokens(self):
        cfg = make_small_cfg(vocab_size=50)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2]])
        out = generate(model, prompt, max_new_tokens=10, context_length=32, temperature=1.0, top_p=0.9)
        generated = out[0, 2:]
        assert (generated >= 0).all()
        assert (generated < 50).all()

    def test_top_p_output_length(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        out = generate(model, prompt, max_new_tokens=5, context_length=32, temperature=1.0, top_p=0.9)
        assert out.shape == (1, 8)

    def test_top_p_tiny_keeps_top_token(self):
        cfg = make_small_cfg()
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2, 3]])
        greedy = generate(model, prompt.clone(), max_new_tokens=5, context_length=32)
        topp = generate(model, prompt.clone(), max_new_tokens=5, context_length=32, temperature=1.0, top_p=0.01)
        assert torch.equal(greedy, topp)

    def test_combined_top_k_and_top_p(self):
        cfg = make_small_cfg(vocab_size=50)
        model = GPTModel(cfg)
        model.eval()
        prompt = torch.tensor([[1, 2]])
        out = generate(
            model, prompt, max_new_tokens=10, context_length=32,
            temperature=0.8, top_k=10, top_p=0.9,
        )
        generated = out[0, 2:]
        assert (generated >= 0).all()
        assert (generated < 50).all()
