import torch
import pytest
from unittest.mock import MagicMock

from minigpt.data import GPTDataset, create_dataloaders


class TestGPTDataset:
    def test_basic_shape(self):
        token_ids = torch.arange(20)
        ds = GPTDataset(token_ids, max_length=4, stride=4)
        x, y = ds[0]
        assert x.shape == (4,)
        assert y.shape == (4,)

    def test_shift_by_one(self):
        token_ids = torch.arange(10)
        ds = GPTDataset(token_ids, max_length=4, stride=4)
        x, y = ds[0]
        assert x.tolist() == [0, 1, 2, 3]
        assert y.tolist() == [1, 2, 3, 4]

    def test_non_overlapping_stride(self):
        token_ids = torch.arange(20)
        ds = GPTDataset(token_ids, max_length=4, stride=4)
        x0, _ = ds[0]
        x1, _ = ds[1]
        assert x0.tolist() == [0, 1, 2, 3]
        assert x1.tolist() == [4, 5, 6, 7]

    def test_overlapping_stride(self):
        token_ids = torch.arange(10)
        ds = GPTDataset(token_ids, max_length=4, stride=2)
        x0, _ = ds[0]
        x1, _ = ds[1]
        assert x0.tolist() == [0, 1, 2, 3]
        assert x1.tolist() == [2, 3, 4, 5]

    def test_length_non_overlapping(self):
        token_ids = torch.arange(20)
        ds = GPTDataset(token_ids, max_length=4, stride=4)
        assert len(ds) == 4

    def test_length_overlapping(self):
        token_ids = torch.arange(10)
        ds = GPTDataset(token_ids, max_length=4, stride=1)
        assert len(ds) == 6

    def test_last_chunk_not_short(self):
        token_ids = torch.arange(20)
        ds = GPTDataset(token_ids, max_length=4, stride=4)
        x, y = ds[len(ds) - 1]
        assert x.shape == (4,)
        assert y.shape == (4,)

    def test_target_is_next_token(self):
        token_ids = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
        ds = GPTDataset(token_ids, max_length=3, stride=3)
        x, y = ds[0]
        for i in range(len(x)):
            assert y[i] == token_ids[i + 1]

    def test_empty_when_too_short(self):
        token_ids = torch.arange(3)
        ds = GPTDataset(token_ids, max_length=4, stride=4)
        assert len(ds) == 0

    def test_single_example(self):
        token_ids = torch.arange(5)
        ds = GPTDataset(token_ids, max_length=4, stride=4)
        assert len(ds) == 1
        x, y = ds[0]
        assert x.tolist() == [0, 1, 2, 3]
        assert y.tolist() == [1, 2, 3, 4]

    def test_stride_one_maximum_overlap(self):
        token_ids = torch.arange(7)
        ds = GPTDataset(token_ids, max_length=4, stride=1)
        assert len(ds) == 3
        for i in range(len(ds)):
            x, y = ds[i]
            assert x.tolist() == list(range(i, i + 4))
            assert y.tolist() == list(range(i + 1, i + 5))

    def test_no_mutation_of_input(self):
        token_ids = torch.arange(10)
        original = token_ids.clone()
        GPTDataset(token_ids, max_length=4, stride=4)
        assert torch.equal(token_ids, original)


class TestCreateDataloaders:
    @staticmethod
    def _make_tokenizer(text_to_ids):
        tok = MagicMock()
        tok.encode = MagicMock(side_effect=lambda t: text_to_ids)
        return tok

    def test_returns_two_loaders(self):
        tok = self._make_tokenizer(list(range(100)))
        train_loader, val_loader = create_dataloaders(
            "dummy", tok, max_length=8, stride=8, batch_size=2
        )
        assert hasattr(train_loader, '__iter__')
        assert hasattr(val_loader, '__iter__')

    def test_train_val_split(self):
        ids = list(range(100))
        tok = self._make_tokenizer(ids)
        train_loader, val_loader = create_dataloaders(
            "dummy", tok, max_length=4, stride=4,
            train_ratio=0.8, batch_size=1, drop_last=False,
        )
        train_count = sum(1 for _ in train_loader)
        val_count = sum(1 for _ in val_loader)
        assert train_count > 0
        assert val_count > 0
        assert train_count > val_count

    def test_batch_shape(self):
        ids = list(range(100))
        tok = self._make_tokenizer(ids)
        train_loader, _ = create_dataloaders(
            "dummy", tok, max_length=8, stride=8, batch_size=4,
        )
        x, y = next(iter(train_loader))
        assert x.shape == (4, 8)
        assert y.shape == (4, 8)

    def test_custom_train_ratio(self):
        ids = list(range(200))
        tok = self._make_tokenizer(ids)
        train_loader, val_loader = create_dataloaders(
            "dummy", tok, max_length=4, stride=4,
            train_ratio=0.5, batch_size=1, drop_last=False,
        )
        train_examples = sum(1 for _ in train_loader)
        val_examples = sum(1 for _ in val_loader)
        assert abs(train_examples - val_examples) <= 2

    def test_drop_last_default(self):
        ids = list(range(50))
        tok = self._make_tokenizer(ids)
        train_loader, _ = create_dataloaders(
            "dummy", tok, max_length=4, stride=4,
            batch_size=3, drop_last=True,
        )
        for x, y in train_loader:
            assert x.shape[0] == 3
