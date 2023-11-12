from functools import partial
from typing import Tuple

import jsonlines
import torch
from torch.utils.data import DataLoader, Dataset


class ContrastivePairsDataset(Dataset):
    def __init__(self, file_path, tokenizer, max_length):
        self.data_pairs = self.load_sentences(file_path)
        self.tokenizer = tokenizer
        self.max_length = max_length

    def load_sentences(self, file_path: str):
        data = []
        with jsonlines.open(file_path, "r") as reader:
            for obj in reader:
                data.append((obj["positive"], obj["negative"]))

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx) -> dict:
        pos, neg = self.data_pairs[idx]
        pos_encoding = self.tokenizer(
            pos,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        neg_encoding = self.tokenizer(
            neg,
            max_length=self.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        return {
            "pos_ids": pos_encoding["input_ids"].squeeze(),
            "pos_mask": pos_encoding["attention_mask"].squeeze(),
            "neg_ids": neg_encoding["input_ids"].squeeze(),
            "neg_mask": neg_encoding["attention_mask"].squeeze(),
        }


def collate_fn(batch: dict) -> Tuple[dict, dict]:
    pos_ids, neg_ids = zip([(item["pos_ids"], item["neg_ids"]) for item in batch])
    pos_mask, neg_mask = zip([(item["pos_mask"], item["neg_mask"]) for item in batch])

    return (
        {
            "input_ids": torch.stack(pos_ids, dim=0),
            "attention_mask": torch.stack(pos_mask, dim=0),
        },
        {
            "input_ids": torch.stack(neg_ids, dim=0),
            "attention_mask": torch.stack(neg_mask, dim=0),
        },
    )


def get_dataloader(data_file: str, batch_size: int, tokenizer, shuffle: bool = True):
    dataset = ContrastivePairsDataset(data_file, tokenizer, max_length=64)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=partial(collate_fn)
    )
