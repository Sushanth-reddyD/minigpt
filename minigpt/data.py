import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    def __init__(self, token_ids, max_length, stride):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - max_length, stride):
            x = token_ids[i : i + max_length]
            y = token_ids[i + 1 : i + max_length + 1]
            self.input_ids.append(x)
            self.target_ids.append(y)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloaders(
    text,
    tokenizer,
    max_length,
    stride,
    train_ratio=0.9,
    batch_size=4,
    shuffle=True,
    drop_last=True,
    num_workers=0,
):
    token_ids = torch.tensor(tokenizer.encode(text))
    split_idx = int(train_ratio * len(token_ids))

    train_data = token_ids[:split_idx]
    val_data = token_ids[split_idx:]

    train_dataset = GPTDataset(train_data, max_length, stride)
    val_dataset = GPTDataset(val_data, max_length, stride)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=drop_last,
        num_workers=num_workers,
    )

    return train_loader, val_loader
