import math
from dataclasses import dataclass
from typing import List, Tuple

import torch
from torch.utils.data import Dataset


@dataclass
class PackedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    sample_indices: torch.Tensor


@dataclass
class PaddedBatch:
    input_ids: torch.Tensor
    labels: torch.Tensor
    attention_mask: torch.Tensor


class TokenizedDataset(Dataset):
    def __init__(self, sequences: List[List[int]]):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq = self.sequences[idx]
        input_ids = seq[:-1]
        labels = seq[1:]
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }


def pack_batch(samples: List[dict], max_tokens: int) -> PackedBatch:
    input_ids = []
    labels = []
    sample_indices = []
    token_count = 0
    sample_id = 0
    for sample in samples:
        seq_len = sample["input_ids"].numel()
        if token_count + seq_len > max_tokens and token_count > 0:
            break
        input_ids.append(sample["input_ids"])
        labels.append(sample["labels"])
        sample_indices.append(torch.full((seq_len,), sample_id, dtype=torch.long))
        token_count += seq_len
        sample_id += 1
    input_ids = torch.cat(input_ids, dim=0)
    labels = torch.cat(labels, dim=0)
    sample_indices = torch.cat(sample_indices, dim=0)
    return PackedBatch(input_ids=input_ids, labels=labels, sample_indices=sample_indices)


def pad_batch(samples: List[dict]) -> PaddedBatch:
    lengths = [sample["input_ids"].numel() for sample in samples]
    max_len = max(lengths)
    batch_size = len(samples)
    input_ids = torch.zeros(batch_size, max_len, dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    attention_mask = torch.zeros(batch_size, max_len, dtype=torch.long)
    for i, sample in enumerate(samples):
        length = sample["input_ids"].numel()
        input_ids[i, :length] = sample["input_ids"]
        labels[i, :length] = sample["labels"]
        attention_mask[i, :length] = 1
    return PaddedBatch(input_ids=input_ids, labels=labels, attention_mask=attention_mask)


def packing_stats(sequences: List[List[int]], max_tokens: int) -> dict:
    lengths = [len(seq) - 1 for seq in sequences]
    avg_len = sum(lengths) / max(1, len(lengths))
    tokens_per_pack = []
    total_padded_tokens = 0
    idx = 0
    while idx < len(lengths):
        tok = 0
        max_len = 0
        count = 0
        while idx < len(lengths) and tok + lengths[idx] <= max_tokens:
            tok += lengths[idx]
            max_len = max(max_len, lengths[idx])
            idx += 1
            count += 1
        tokens_per_pack.append(tok)
        total_padded_tokens += max_len * count
    saved = 1.0 - (sum(tokens_per_pack) / max(1, total_padded_tokens))
    return {
        "avg_seq_len": avg_len,
        "tokens_per_packed_batch": sum(tokens_per_pack) / max(1, len(tokens_per_pack)),
        "pct_saved_vs_padding": saved * 100.0,
    }
