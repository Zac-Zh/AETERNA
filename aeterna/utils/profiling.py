"""Profiling utilities for AETERNA training and inference."""
import contextlib
import time
from typing import Dict, Optional

import torch


class Timer:
    """Simple context manager for timing code blocks."""

    def __init__(self, name: str = "block", verbose: bool = True):
        self.name = name
        self.verbose = verbose
        self.elapsed = 0.0

    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *args):
        self.elapsed = time.perf_counter() - self.start
        if self.verbose:
            print(f"{self.name}: {self.elapsed:.4f}s")


class ProfilerContext:
    """Context manager for PyTorch profiler with optional export to Chrome trace."""

    def __init__(
        self,
        enabled: bool = True,
        wait: int = 1,
        warmup: int = 1,
        active: int = 3,
        repeat: int = 1,
        output_dir: str = "./profiler_logs",
        record_shapes: bool = True,
        with_stack: bool = False,
    ):
        self.enabled = enabled
        self.wait = wait
        self.warmup = warmup
        self.active = active
        self.repeat = repeat
        self.output_dir = output_dir
        self.record_shapes = record_shapes
        self.with_stack = with_stack
        self.profiler: Optional[torch.profiler.profile] = None

    def __enter__(self):
        if not self.enabled:
            return contextlib.nullcontext()

        self.profiler = torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(
                wait=self.wait,
                warmup=self.warmup,
                active=self.active,
                repeat=self.repeat,
            ),
            on_trace_ready=torch.profiler.tensorboard_trace_handler(self.output_dir),
            record_shapes=self.record_shapes,
            with_stack=self.with_stack,
        )
        self.profiler.__enter__()
        return self

    def __exit__(self, *args):
        if self.profiler is not None:
            self.profiler.__exit__(*args)

    def step(self):
        """Call this after each training step when profiling."""
        if self.profiler is not None:
            self.profiler.step()


def memory_summary(device: Optional[torch.device] = None) -> Dict[str, float]:
    """Get CUDA memory statistics in MB.

    Args:
        device: CUDA device to query (default: current device)

    Returns:
        Dictionary with memory stats in MB
    """
    if not torch.cuda.is_available():
        return {}

    if device is None:
        device = torch.cuda.current_device()

    return {
        "allocated_mb": torch.cuda.memory_allocated(device) / 1024**2,
        "reserved_mb": torch.cuda.memory_reserved(device) / 1024**2,
        "max_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
        "max_reserved_mb": torch.cuda.max_memory_reserved(device) / 1024**2,
    }


def reset_peak_memory_stats(device: Optional[torch.device] = None):
    """Reset peak memory stats for profiling."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)


@contextlib.contextmanager
def profile_memory(name: str = "block", verbose: bool = True):
    """Context manager to profile memory usage of a code block.

    Args:
        name: Name of the block for logging
        verbose: Whether to print results
    """
    if not torch.cuda.is_available():
        yield
        return

    reset_peak_memory_stats()
    yield

    stats = memory_summary()
    if verbose:
        print(f"{name} memory: allocated={stats['allocated_mb']:.1f}MB "
              f"peak={stats['max_allocated_mb']:.1f}MB")
