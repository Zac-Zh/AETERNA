"""Normalization layers for AETERNA models."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    More efficient than LayerNorm, commonly used in modern LLMs.
    Normalizes using RMS without centering (no mean subtraction).
    """

    def __init__(self, d_model: int, eps: float = 1e-6):
        """Initialize RMSNorm.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
        """
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply RMS normalization.

        Args:
            x: Input tensor of shape (..., d_model)

        Returns:
            Normalized tensor of same shape
        """
        # Compute RMS: sqrt(mean(x^2))
        rms = torch.sqrt(torch.mean(x ** 2, dim=-1, keepdim=True) + self.eps)

        # Normalize and scale
        x_normed = x / rms
        return self.weight * x_normed


class LayerNorm(nn.LayerNorm):
    """Standard LayerNorm with optional bias control."""

    def __init__(self, d_model: int, eps: float = 1e-6, bias: bool = True):
        """Initialize LayerNorm.

        Args:
            d_model: Model dimension
            eps: Small constant for numerical stability
            bias: Whether to include bias parameter
        """
        super().__init__(d_model, eps=eps, elementwise_affine=True)
        if not bias and hasattr(self, 'bias') and self.bias is not None:
            self.register_parameter('bias', None)
