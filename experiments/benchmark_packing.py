import argparse
import time
import yaml

import torch

from aeterna.data.loader import build_datasets, packed_loader, padded_loader
from aeterna.model.aeterna_model import AeternaLM


def benchmark_packing(config: dict) -> None:
    dataset, _ = build_datasets(
        dataset_name=config["dataset_name"],
        dataset_config=config["dataset_config"],
        tokenizer_name=config["tokenizer_name"],
        max_length=config["max_length"],
        seed=config["seed"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeternaLM(vocab_size=50257).to(device)

    packed = packed_loader(
        dataset,
        max_tokens=config["max_tokens"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    padded = padded_loader(dataset, batch_size=config["batch_size"], seed=config["seed"])

    def run(loader, kind: str, steps: int = 50):
        total_tokens = 0
        start = time.time()
        for i, batch in enumerate(loader):
            if i >= steps:
                break
            if kind == "packed":
                input_ids = batch.input_ids.to(device)
                sample_indices = batch.sample_indices.to(device)
                logits, _ = model.forward_packed(input_ids, sample_indices)
                total_tokens += input_ids.numel()
            else:
                input_ids = batch.input_ids.to(device)
                attention_mask = batch.attention_mask.to(device)
                logits, _ = model.forward_padded(input_ids, attention_mask)
                total_tokens += attention_mask.sum().item()
            _ = logits.mean()
        elapsed = time.time() - start
        return total_tokens / max(elapsed, 1e-6)

    packed_tps = run(packed, "packed")
    padded_tps = run(padded, "padded")
    utilization = packed_tps / max(padded_tps, 1e-6)
    print(f"Packed tokens/sec: {packed_tps:.1f}")
    print(f"Padded tokens/sec: {padded_tps:.1f}")
    print(f"Utilization (packed/padded): {utilization:.2f}x")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aeterna_small.yaml")
    args = parser.parse_args()
    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    benchmark_packing(config)
