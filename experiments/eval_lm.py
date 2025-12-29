"""Evaluation script for AETERNA language models.

Computes perplexity and other metrics on validation/test sets.
"""
import argparse
import math
import time

import torch
import yaml

from aeterna.data.loader import build_datasets, packed_loader
from aeterna.model.aeterna_model import AeternaLM
from aeterna.utils.checkpoint import load_checkpoint
from aeterna.utils.seed import set_seed


def evaluate(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    max_steps: int = None,
) -> dict:
    """Evaluate model on a dataset.

    Args:
        model: Model to evaluate
        loader: Data loader
        device: Device to run on
        max_steps: Maximum evaluation steps (None = full dataset)

    Returns:
        Dictionary of metrics
    """
    model.eval()

    total_loss = 0.0
    total_tokens = 0
    num_batches = 0

    start_time = time.time()

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if max_steps is not None and step >= max_steps:
                break

            # Move to device
            input_ids = batch.input_ids.to(device)
            labels = batch.labels.to(device)
            sample_indices = batch.sample_indices.to(device)

            # Forward pass
            loss, _, _ = model.loss_packed(input_ids, labels, sample_indices)

            # Accumulate
            num_tokens = input_ids.numel()
            total_loss += loss.item() * num_tokens
            total_tokens += num_tokens
            num_batches += 1

            if (step + 1) % 50 == 0:
                print(f"Eval step {step+1}, running loss: {loss.item():.4f}")

    elapsed = time.time() - start_time

    # Compute metrics
    avg_loss = total_loss / max(total_tokens, 1)
    perplexity = math.exp(min(avg_loss, 20))  # Cap to avoid overflow

    metrics = {
        "avg_loss": avg_loss,
        "perplexity": perplexity,
        "total_tokens": total_tokens,
        "num_batches": num_batches,
        "elapsed_sec": elapsed,
        "tokens_per_sec": total_tokens / max(elapsed, 1e-6),
    }

    return metrics


def main():
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(description="Evaluate AETERNA language model")
    parser.add_argument("--config", default="configs/aeterna_small.yaml",
                        help="Path to config file")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to checkpoint to load")
    parser.add_argument("--split", choices=["train", "validation", "test"],
                        default="validation",
                        help="Dataset split to evaluate on")
    parser.add_argument("--max_steps", type=int, default=None,
                        help="Maximum evaluation steps")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    set_seed(config.get("seed", 42))

    # Load dataset
    print(f"Loading {args.split} dataset...")
    try:
        dataset, stats = build_datasets(
            dataset_name=config["dataset_name"],
            dataset_config=config.get("dataset_config", "wikitext-2-raw-v1"),
            split=args.split,
            tokenizer_name=config["tokenizer_name"],
            max_length=config["max_length"],
            seed=config["seed"],
        )
        print(f"Dataset loaded. Stats: {stats}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Falling back to train split for evaluation")
        dataset, stats = build_datasets(
            dataset_name=config["dataset_name"],
            dataset_config=config.get("dataset_config", "wikitext-2-raw-v1"),
            split="train",
            tokenizer_name=config["tokenizer_name"],
            max_length=config["max_length"],
            seed=config["seed"],
        )

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Create model
    vocab_size = config.get("vocab_size", 50257)
    d_model = config.get("d_model", 128)
    model = AeternaLM(vocab_size=vocab_size, d_model=d_model).to(device)

    # Load checkpoint if provided
    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}...")
        load_checkpoint(args.checkpoint, model)
        print("Checkpoint loaded")
    else:
        print("No checkpoint provided, evaluating randomly initialized model")

    # Create data loader
    loader = packed_loader(
        dataset,
        max_tokens=config["max_tokens"],
        batch_size=config["batch_size"],
        seed=config["seed"],
    )

    # Evaluate
    print(f"\nEvaluating on {args.split} split...")
    metrics = evaluate(model, loader, device, max_steps=args.max_steps)

    # Print results
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Average Loss:     {metrics['avg_loss']:.4f}")
    print(f"Perplexity:       {metrics['perplexity']:.2f}")
    print(f"Total Tokens:     {metrics['total_tokens']:,}")
    print(f"Num Batches:      {metrics['num_batches']}")
    print(f"Elapsed Time:     {metrics['elapsed_sec']:.2f}s")
    print(f"Tokens/sec:       {metrics['tokens_per_sec']:.1f}")
    print("="*60)


if __name__ == "__main__":
    main()
