"""Mamba-2 / SSD-style block implementation.

Implements a simplified Mamba-2 block with:
- Selective State Space Model (SSM) layer
- Support for variable-length packed sequences
- State reset at sequence boundaries
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from aeterna.ops.varlen_scan import MambaVarlenScan
from aeterna.layers.norms import RMSNorm


class Mamba2Block(nn.Module):
    """Mamba-2 / SSD-style block with variable-length support.

    Architecture:
        x -> norm -> [linear -> SSM -> linear] -> residual -> output

    Supports both packed (with sample_indices) and padded (with mask) inputs.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize Mamba2 block.

        Args:
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size (not used in current simplified version)
            expand: Expansion factor for intermediate dimension
            dropout: Dropout probability
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = d_model * expand

        # Pre-norm
        self.norm = RMSNorm(d_model)

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner, bias=False)

        # SSM core
        self.ssm = MambaVarlenScan(self.d_inner, d_state)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward_packed(
        self,
        x: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass for packed sequences.

        Args:
            x: (T, D) packed input tokens
            sample_indices: (T,) sequence ID for each token

        Returns:
            output: (T, D) after Mamba block with residual
        """
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Input projection
        x = self.in_proj(x)

        # Apply SSM with boundary resets
        x, _ = self.ssm(x, sample_indices)

        # Output projection
        x = self.out_proj(x)
        x = self.dropout(x)

        # Residual connection
        return residual + x

    def forward_padded(
        self,
        x: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass for padded batch.

        Args:
            x: (B, L, D) padded input
            attention_mask: (B, L) mask where 1=valid, 0=padding

        Returns:
            output: (B, L, D) after Mamba block with residual
        """
        B, L, D = x.shape
        residual = x

        # Pre-norm
        x = self.norm(x)

        # Flatten to (B*L, D) for processing
        x_flat = x.reshape(B * L, D)

        # Create sample indices for each sequence in the batch
        # Each sequence gets a unique ID
        sample_indices = torch.arange(B, device=x.device).repeat_interleave(L)

        # Input projection
        x_flat = self.in_proj(x_flat)

        # Apply SSM
        x_flat, _ = self.ssm(x_flat, sample_indices)

        # Output projection
        x_flat = self.out_proj(x_flat)

        # Reshape back
        x = x_flat.reshape(B, L, D)
        x = self.dropout(x)

        # Apply mask if provided
        if attention_mask is not None:
            x = x * attention_mask.unsqueeze(-1)

        # Residual connection
        return residual + x

    def forward(
        self,
        x: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with automatic mode detection.

        Args:
            x: Input tensor, either (T, D) packed or (B, L, D) padded
            sample_indices: (T,) for packed mode
            attention_mask: (B, L) for padded mode

        Returns:
            output: Same shape as input
        """
        if sample_indices is not None:
            # Packed mode
            assert x.ndim == 2, "Packed input must be (T, D)"
            return self.forward_packed(x, sample_indices)
        else:
            # Padded mode
            assert x.ndim == 3, "Padded input must be (B, L, D)"
            return self.forward_padded(x, attention_mask)


class Mamba2Stack(nn.Module):
    """Stack of Mamba2 blocks forming a deep SSM backbone."""

    def __init__(
        self,
        num_layers: int,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.0,
    ):
        """Initialize Mamba2 stack.

        Args:
            num_layers: Number of Mamba blocks
            d_model: Model dimension
            d_state: SSM state dimension
            d_conv: Convolution kernel size
            expand: Expansion factor
            dropout: Dropout probability
        """
        super().__init__()
        self.layers = nn.ModuleList([
            Mamba2Block(d_model, d_state, d_conv, expand, dropout)
            for _ in range(num_layers)
        ])
        self.norm = RMSNorm(d_model)

    def forward(
        self,
        x: torch.Tensor,
        sample_indices: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward through all layers.

        Args:
            x: Input tensor
            sample_indices: Optional sample indices for packed mode
            attention_mask: Optional mask for padded mode

        Returns:
            output: Processed tensor
        """
        for layer in self.layers:
            x = layer(x, sample_indices, attention_mask)

        return self.norm(x)
