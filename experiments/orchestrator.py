"""Orchestrator for AETERNA experiments.

Provides CLI for training, evaluation, benchmarking, and sweeps.
"""
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List

import torch
import yaml

from aeterna.data.loader import build_datasets, packed_loader
from aeterna.loss.vnge_loss import vnge_loss
from aeterna.model.aeterna_model import AeternaLM
from aeterna.utils.checkpoint import save_checkpoint, load_checkpoint
from aeterna.utils.logging import log_jsonl
from aeterna.utils.seed import set_seed


def train_run(
    config: Dict,
    lambda_vnge: float,
    output_dir: str = "logs",
    checkpoint_dir: str = "checkpoints",
) -> Dict:
    """Run a single training run.

    Args:
        config: Configuration dictionary
        lambda_vnge: VNGE regularization weight
        output_dir: Directory for logs
        checkpoint_dir: Directory for checkpoints

    Returns:
        Summary statistics
    """
    set_seed(config["seed"])

    # Load dataset
    print(f"Loading dataset: {config['dataset_name']} ...")
    dataset, stats = build_datasets(
        dataset_name=config["dataset_name"],
        dataset_config=config.get("dataset_config", "wikitext-2-raw-v1"),
        tokenizer_name=config["tokenizer_name"],
        max_length=config["max_length"],
        seed=config["seed"],
    )
    print(f"Dataset loaded. Stats: {stats}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    vocab_size = config.get("vocab_size", 50257)  # GPT-2 vocab size
    d_model = config.get("d_model", 128)
    model = AeternaLM(vocab_size=vocab_size, d_model=d_model).to(device)

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")

    # Optimizer
    lr = config["learning_rate"]
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    # Create data loader
    loader = packed_loader(
        dataset,
        max_tokens=config["max_tokens"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )

    # Setup logging
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    log_path = f"{output_dir}/train_lambda_{lambda_vnge}.jsonl"

    # Training loop
    step = 0
    total_tokens = 0
    start_time = time.time()
    loss_history = []

    print(f"\nTraining for {config['steps']} steps with lambda_vnge={lambda_vnge}...")

    for batch in loader:
        model.train()

        # Move to device
        input_ids = batch.input_ids.to(device)
        labels = batch.labels.to(device)
        sample_indices = batch.sample_indices.to(device)

        # Forward pass
        task_loss, _, hidden = model.loss_packed(input_ids, labels, sample_indices)

        # VNGE regularization
        vnge_value = torch.tensor(0.0, device=device)
        if lambda_vnge > 0:
            vnge_value, _ = vnge_loss(
                hidden,
                sample_indices,
                window=config.get("vnge_window", 8),
                num_probes=config.get("vnge_probes", 4),
                cheb_order=config.get("vnge_cheb_order", 8),
            )

        # Total loss
        total_loss = task_loss + lambda_vnge * vnge_value

        # Check for NaN/Inf
        if not torch.isfinite(total_loss):
            print(f"WARNING: Non-finite loss at step {step}. Skipping update.")
            continue

        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        # Update
        optimizer.step()

        # Track metrics
        num_tokens = input_ids.numel()
        total_tokens += num_tokens
        loss_history.append(task_loss.item())

        # Logging
        if step % config.get("log_interval", 20) == 0:
            elapsed = time.time() - start_time
            tokens_per_sec = total_tokens / max(elapsed, 1e-6)

            log_entry = {
                "step": step,
                "task_loss": task_loss.item(),
                "vnge_loss": vnge_value.item(),
                "total_loss": total_loss.item(),
                "tokens_per_sec": tokens_per_sec,
                "lambda_vnge": lambda_vnge,
                "num_tokens": num_tokens,
            }
            log_jsonl(log_path, log_entry)

            print(
                f"Step {step:4d} | "
                f"task_loss={task_loss.item():.4f} | "
                f"vnge={vnge_value.item():.4f} | "
                f"total={total_loss.item():.4f} | "
                f"tok/s={tokens_per_sec:.1f}"
            )

        # Checkpointing
        if (step + 1) % config.get("checkpoint_interval", 500) == 0:
            ckpt_path = f"{checkpoint_dir}/step_{step+1}_lambda_{lambda_vnge}.pt"
            save_checkpoint(ckpt_path, model, optimizer, step)
            print(f"Saved checkpoint: {ckpt_path}")

        step += 1
        if step >= config["steps"]:
            break

    # Final summary
    elapsed = time.time() - start_time
    avg_loss = sum(loss_history) / max(len(loss_history), 1)

    summary = {
        "steps": step,
        "avg_task_loss": avg_loss,
        "final_task_loss": loss_history[-1] if loss_history else 0.0,
        "total_tokens": total_tokens,
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-6),
        "lambda_vnge": lambda_vnge,
    }

    print(f"\nTraining complete!")
    print(f"Average loss: {avg_loss:.4f}")
    print(f"Final loss: {summary['final_task_loss']:.4f}")
    print(f"Tokens/sec: {summary['tokens_per_sec']:.1f}")

    return summary


def sweep_lambda(config: Dict, output_dir: str = "logs/sweeps"):
    """Run lambda sweep experiment.

    Args:
        config: Configuration with lambda_vnge_values list
        output_dir: Output directory for sweep results
    """
    lambda_values = config.get("lambda_vnge_values", [0.0, 0.1])

    print(f"\n{'='*60}")
    print(f"Running lambda sweep with values: {lambda_values}")
    print(f"{'='*60}\n")

    results = []
    for lam in lambda_values:
        print(f"\n--- Lambda = {lam} ---")
        summary = train_run(config, lambda_vnge=lam, output_dir=output_dir)
        summary["lambda"] = lam
        results.append(summary)

    # Save sweep summary
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    summary_path = f"{output_dir}/lambda_sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSweep complete! Results saved to {summary_path}")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(description="AETERNA experiment orchestrator")
    parser.add_argument("--config", default="configs/aeterna_small.yaml",
                        help="Path to config file")
    parser.add_argument("--mode", choices=["train", "sweep"], default="train",
                        help="Experiment mode")
    parser.add_argument("--lambda_vnge", type=float, default=None,
                        help="VNGE lambda (overrides config)")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    print(f"Loaded config from {args.config}")
    print(f"Mode: {args.mode}")

    if args.mode == "train":
        # Single training run
        lambda_vnge = args.lambda_vnge if args.lambda_vnge is not None else \
                      config.get("lambda_vnge_values", [0.0])[0]
        train_run(config, lambda_vnge=lambda_vnge)

    elif args.mode == "sweep":
        # Lambda sweep
        sweep_lambda(config)


if __name__ == "__main__":
    main()
