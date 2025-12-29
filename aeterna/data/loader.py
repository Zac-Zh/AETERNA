import random
from typing import Iterator, List, Tuple

import torch
from datasets import load_dataset
from transformers import AutoTokenizer

from aeterna.data.packing import TokenizedDataset, pack_batch, pad_batch, packing_stats
from aeterna.utils.seed import set_seed


def load_text_dataset(name: str, split: str = "train") -> List[str]:
    dataset = load_dataset(name, split=split)
    if "text" in dataset.features:
        return [row["text"] for row in dataset]
    if "content" in dataset.features:
        return [row["content"] for row in dataset]
    raise ValueError("Dataset does not have a text field")


def tokenize_texts(texts: List[str], tokenizer, max_length: int = 256) -> List[List[int]]:
    sequences = []
    for text in texts:
        tokens = tokenizer.encode(text, truncation=True, max_length=max_length)
        if len(tokens) > 1:
            sequences.append(tokens)
    return sequences


def build_datasets(
    dataset_name: str = "wikitext",
    dataset_config: str = "wikitext-2-raw-v1",
    split: str = "train",
    tokenizer_name: str = "gpt2",
    max_length: int = 256,
    seed: int = 42,
) -> Tuple[TokenizedDataset, dict]:
    set_seed(seed)
    if dataset_name == "wikitext":
        texts = load_dataset(dataset_name, dataset_config, split=split)
        texts = [row["text"] for row in texts]
    else:
        texts = load_text_dataset(dataset_name, split=split)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    sequences = tokenize_texts(texts, tokenizer, max_length=max_length)
    stats = packing_stats(sequences, max_tokens=max_length * 4)
    return TokenizedDataset(sequences), stats


def packed_loader(
    dataset: TokenizedDataset,
    max_tokens: int,
    batch_size: int,
    seed: int = 42,
) -> Iterator:
    set_seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    idx = 0
    while idx < len(indices):
        batch_indices = indices[idx : idx + batch_size]
        samples = [dataset[i] for i in batch_indices]
        packed = pack_batch(samples, max_tokens=max_tokens)
        idx += batch_size
        yield packed


def padded_loader(dataset: TokenizedDataset, batch_size: int, seed: int = 42) -> Iterator:
    set_seed(seed)
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    idx = 0
    while idx < len(indices):
        batch_indices = indices[idx : idx + batch_size]
        samples = [dataset[i] for i in batch_indices]
        idx += batch_size
        yield pad_batch(samples)
