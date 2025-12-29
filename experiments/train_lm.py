import argparse
import time
import yaml

import torch

from aeterna.data.loader import build_datasets, packed_loader
from aeterna.loss.vnge_loss import vnge_loss
from aeterna.model.aeterna_model import AeternaLM
from aeterna.utils.logging import log_jsonl
from aeterna.utils.seed import set_seed


def run_training(config: dict, lambda_vnge: float) -> None:
    set_seed(config["seed"])
    dataset, stats = build_datasets(
        dataset_name=config["dataset_name"],
        dataset_config=config["dataset_config"],
        tokenizer_name=config["tokenizer_name"],
        max_length=config["max_length"],
        seed=config["seed"],
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AeternaLM(vocab_size=50257).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])

    loader = packed_loader(
        dataset,
        max_tokens=config["max_tokens"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )
    log_path = f"logs/train_lambda_{lambda_vnge}.jsonl"

    step = 0
    start_time = time.time()
    for batch in loader:
        model.train()
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        sample_indices = batch.sample_indices.to(device)
        task_loss, _, hidden = model.loss_packed(input_ids, labels, sample_indices)
        vnge_value = torch.tensor(0.0, device=device)
        if lambda_vnge > 0:
            vnge_value, _ = vnge_loss(
                hidden,
                sample_indices,
                window=config["vnge_window"],
                num_probes=config["vnge_probes"],
                cheb_order=config["vnge_cheb_order"],
            )
        total_loss = task_loss + lambda_vnge * vnge_value
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        elapsed = time.time() - start_time
        tokens = input_ids.numel()
        tokens_per_sec = tokens / max(elapsed, 1e-6)
        log_jsonl(
            log_path,
            {
                "step": step,
                "task_loss": task_loss.item(),
                "vnge_loss": vnge_value.item(),
                "total_loss": total_loss.item(),
                "tokens_per_sec": tokens_per_sec,
                "packing_stats": stats,
                "lambda_vnge": lambda_vnge,
            },
        )
        if step % 20 == 0:
            print(
                f"step {step} task_loss={task_loss.item():.4f} vnge={vnge_value.item():.4f} "
                f"total={total_loss.item():.4f} tok/s={tokens_per_sec:.1f}"
            )
        step += 1
        start_time = time.time()
        if step >= config["steps"]:
            break


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/aeterna_small.yaml")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    for lambda_vnge in config.get("lambda_vnge_values", [0.0]):
        run_training(config, float(lambda_vnge))


if __name__ == "__main__":
    main()
