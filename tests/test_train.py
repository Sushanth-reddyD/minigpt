import math
import torch
from torch.utils.data import DataLoader

from minigpt.model import GPTModel
from minigpt.data import GPTDataset
from minigpt.train import calc_loss_batch, calc_loss_loader, get_lr, train_model

TINY_CONFIG = {
    "vocab_size": 32,
    "context_length": 16,
    "emb_dim": 32,
    "n_heads": 2,
    "n_layers": 2,
    "drop_rate": 0.0,
    "qkv_bias": False,
}


def _make_loader(num_tokens=100, max_length=16, batch_size=4):
    token_ids = torch.randint(0, TINY_CONFIG["vocab_size"], (num_tokens,))
    dataset = GPTDataset(token_ids, max_length=max_length, stride=max_length)
    return DataLoader(dataset, batch_size=batch_size, drop_last=True)


# ── calc_loss_batch ──────────────────────────────────────────


def test_calc_loss_batch_returns_scalar():
    model = GPTModel(TINY_CONFIG)
    loader = _make_loader()
    input_ids, target_ids = next(iter(loader))
    loss = calc_loss_batch(input_ids, target_ids, model, "cpu")
    assert loss.dim() == 0


def test_calc_loss_batch_is_positive():
    model = GPTModel(TINY_CONFIG)
    loader = _make_loader()
    input_ids, target_ids = next(iter(loader))
    loss = calc_loss_batch(input_ids, target_ids, model, "cpu")
    assert loss.item() > 0


def test_calc_loss_batch_random_model_finite():
    torch.manual_seed(42)
    model = GPTModel(TINY_CONFIG)
    loader = _make_loader(num_tokens=500, batch_size=16)
    for input_ids, target_ids in loader:
        loss = calc_loss_batch(input_ids, target_ids, model, "cpu")
        assert torch.isfinite(loss)


def test_calc_loss_batch_requires_grad():
    model = GPTModel(TINY_CONFIG)
    loader = _make_loader()
    input_ids, target_ids = next(iter(loader))
    loss = calc_loss_batch(input_ids, target_ids, model, "cpu")
    assert loss.requires_grad


# ── calc_loss_loader ─────────────────────────────────────────


def test_calc_loss_loader_returns_float():
    model = GPTModel(TINY_CONFIG)
    loader = _make_loader()
    loss = calc_loss_loader(loader, model, "cpu")
    assert isinstance(loss, float)


def test_calc_loss_loader_num_batches_cap():
    model = GPTModel(TINY_CONFIG)
    loader = _make_loader(num_tokens=200)
    loss_all = calc_loss_loader(loader, model, "cpu")
    loss_2 = calc_loss_loader(loader, model, "cpu", num_batches=2)
    assert isinstance(loss_2, float)
    assert loss_2 > 0


def test_calc_loss_loader_num_batches_exceeds_loader():
    model = GPTModel(TINY_CONFIG)
    loader = _make_loader(num_tokens=80)
    n = len(loader)
    loss_exact = calc_loss_loader(loader, model, "cpu", num_batches=n)
    loss_over = calc_loss_loader(loader, model, "cpu", num_batches=n + 100)
    assert abs(loss_exact - loss_over) < 1e-5


def test_calc_loss_loader_eval_mode_no_grad():
    model = GPTModel(TINY_CONFIG)
    model.eval()
    loader = _make_loader()
    with torch.no_grad():
        loss = calc_loss_loader(loader, model, "cpu")
    assert loss > 0


# ── get_lr ───────────────────────────────────────────────────


def test_get_lr_starts_at_zero():
    lr = get_lr(step=0, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
    assert lr == 0.0


def test_get_lr_warmup_midpoint():
    lr = get_lr(step=5, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
    assert abs(lr - 2.5e-4) < 1e-8


def test_get_lr_peak_at_warmup_end():
    lr = get_lr(step=10, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
    assert abs(lr - 5e-4) < 1e-8


def test_get_lr_reaches_min_at_max_steps():
    lr = get_lr(step=100, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
    assert abs(lr - 5e-5) < 1e-8


def test_get_lr_cosine_midpoint():
    lr = get_lr(step=55, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
    expected = 5e-5 + 0.5 * (5e-4 - 5e-5) * (1.0 + math.cos(math.pi * 0.5))
    assert abs(lr - expected) < 1e-10


def test_get_lr_beyond_max_steps_returns_min():
    lr = get_lr(step=150, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
    assert abs(lr - 5e-5) < 1e-8


def test_get_lr_monotonically_decreasing_after_warmup():
    lrs = [
        get_lr(step=s, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
        for s in range(10, 101)
    ]
    for i in range(1, len(lrs)):
        assert lrs[i] <= lrs[i - 1] + 1e-10


def test_get_lr_monotonically_increasing_during_warmup():
    lrs = [
        get_lr(step=s, warmup_steps=10, max_steps=100, max_lr=5e-4, min_lr=5e-5)
        for s in range(11)
    ]
    for i in range(1, len(lrs)):
        assert lrs[i] >= lrs[i - 1]


# ── train_model ──────────────────────────────────────────────


class _FakeTokenizer:
    def encode(self, text):
        return [0, 1, 2]

    def decode(self, ids):
        return "fake output"


def test_train_model_returns_dict_with_keys():
    model = GPTModel(TINY_CONFIG)
    train_loader = _make_loader(num_tokens=100)
    val_loader = _make_loader(num_tokens=60)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device="cpu",
        num_epochs=1,
        eval_freq=5,
        eval_iter=2,
        start_context="test",
        tokenizer=_FakeTokenizer(),
        warmup_steps=2,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    assert "train_losses" in result
    assert "val_losses" in result
    assert "lrs" in result


def test_train_model_loss_decreases():
    torch.manual_seed(42)
    model = GPTModel(TINY_CONFIG)
    train_loader = _make_loader(num_tokens=500)
    val_loader = _make_loader(num_tokens=200)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device="cpu",
        num_epochs=5,
        eval_freq=5,
        eval_iter=2,
        start_context="test",
        tokenizer=_FakeTokenizer(),
        warmup_steps=2,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    assert len(result["train_losses"]) >= 2
    assert result["train_losses"][-1] < result["train_losses"][0]


def test_train_model_lr_tracking():
    model = GPTModel(TINY_CONFIG)
    train_loader = _make_loader(num_tokens=500)
    val_loader = _make_loader(num_tokens=200)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    result = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device="cpu",
        num_epochs=2,
        eval_freq=100,
        eval_iter=2,
        start_context="test",
        tokenizer=_FakeTokenizer(),
        warmup_steps=2,
        max_lr=1e-3,
        min_lr=1e-4,
    )

    total_steps = 2 * len(train_loader)
    assert len(result["lrs"]) == total_steps
    assert result["lrs"][0] == 0.0
    assert max(result["lrs"]) > 0


def test_train_model_parameters_updated():
    torch.manual_seed(42)
    model = GPTModel(TINY_CONFIG)
    train_loader = _make_loader(num_tokens=500)
    val_loader = _make_loader(num_tokens=200)

    params_before = {
        name: p.clone() for name, p in model.named_parameters()
    }

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.1)

    train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        device="cpu",
        num_epochs=2,
        eval_freq=100,
        eval_iter=2,
        start_context="test",
        tokenizer=_FakeTokenizer(),
        warmup_steps=2,
        max_lr=1e-3,
        min_lr=1e-4,
        max_norm=1.0,
    )

    changed = 0
    for name, p in model.named_parameters():
        if not torch.equal(p, params_before[name]):
            changed += 1
    assert changed > 0
