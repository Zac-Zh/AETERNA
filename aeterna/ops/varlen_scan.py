"""Variable-length packed scan operations with state resets.

Implements PackMamba-style processing where hidden states are reset
at sequence boundaries in packed batches.
"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .triton_kernels import has_triton, triton_varlen_scan
from .pack_reset import reset_state_on_boundaries


class MambaVarlenScan(nn.Module):
    """Variable-length Mamba scan with boundary resets.

    Implements a simplified SSM recurrence compatible with packed sequences.
    State is automatically reset when crossing sequence boundaries.
    """

    def __init__(self, d_model: int, d_state: int = 16):
        """Initialize Mamba varlen scan.

        Args:
            d_model: Model dimension
            d_state: Hidden state dimension for SSM
        """
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        # SSM parameters (simplified)
        # A: state transition matrix (d_state,)
        # B: input projection (d_model -> d_state)
        # C: output projection (d_state -> d_model)
        # D: skip connection weight
        self.A = nn.Parameter(torch.randn(d_state))
        self.B = nn.Linear(d_model, d_state, bias=False)
        self.C = nn.Linear(d_state, d_model, bias=False)
        self.D = nn.Parameter(torch.ones(1))

        # Initialize A to be stable (negative eigenvalues)
        nn.init.uniform_(self.A, -1.0, -0.1)

    def forward_reference(
        self,
        x: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Reference implementation using PyTorch loops.

        This is the ground truth for correctness testing.

        Args:
            x: (T, D) packed input tokens
            sample_indices: (T,) sequence ID for each token

        Returns:
            y: (T, D) output
            final_states: None (not implemented yet)
        """
        T, D = x.shape
        device = x.device
        dtype = x.dtype

        # Get reset mask (True where state should be reset)
        reset_mask = reset_state_on_boundaries(None, sample_indices)

        # Initialize state
        h = torch.zeros(self.d_state, device=device, dtype=dtype)
        outputs = torch.zeros(T, D, device=device, dtype=dtype)

        # SSM discretization (simplified Euler)
        # In practice, would use more sophisticated discretization
        dt = 0.1  # timestep

        for t in range(T):
            # Reset state at boundaries
            if reset_mask[t]:
                h = torch.zeros_like(h)

            # SSM recurrence:
            # h_t = (1 + dt * A) * h_{t-1} + dt * B * x_t
            # y_t = C * h_t + D * x_t
            x_t = x[t]
            b_t = self.B(x_t)  # (d_state,)

            # Discretized state update
            h = (1.0 + dt * self.A) * h + dt * b_t

            # Output computation
            c_t = self.C(h)  # (d_model,)
            y_t = c_t + self.D * x_t

            outputs[t] = y_t

        return outputs, None

    def forward_optimized(
        self,
        x: torch.Tensor,
        sample_indices: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Optimized implementation using Triton or torch.compile.

        Falls back to reference if optimization not available.

        Args:
            x: (T, D) packed input tokens
            sample_indices: (T,) sequence ID for each token

        Returns:
            y: (T, D) output
            final_states: None
        """
        if has_triton():
            # Use Triton kernel (currently just a placeholder)
            # Full implementation would pass SSM parameters to kernel
            # For now, fall back to reference
            return self.forward_reference(x, sample_indices)
        else:
            # Try torch.compile if available (Torch 2.0+)
            try:
                compiled_fn = torch.compile(self.forward_reference)
                return compiled_fn(x, sample_indices)
            except:
                # Fall back to reference
                return self.forward_reference(x, sample_indices)

    def forward(
        self,
        x: torch.Tensor,
        sample_indices: torch.Tensor,
        use_reference: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Forward pass with automatic optimization selection.

        Args:
            x: (T, D) packed input tokens
            sample_indices: (T,) sequence ID for each token
            use_reference: Force use of reference implementation

        Returns:
            y: (T, D) output
            final_states: Optional per-sequence final states
        """
        if use_reference:
            return self.forward_reference(x, sample_indices)
        else:
            return self.forward_optimized(x, sample_indices)


def mamba_varlen_scan(
    x: torch.Tensor,
    sample_indices: torch.Tensor,
    ssm_module: Optional[MambaVarlenScan] = None,
    use_reference: bool = False,
) -> torch.Tensor:
    """Functional interface for variable-length Mamba scan.

    Args:
        x: (T, D) packed input tokens
        sample_indices: (T,) sequence ID for each token
        ssm_module: Optional pre-initialized SSM module
        use_reference: Whether to use reference implementation

    Returns:
        y: (T, D) output after SSM processing with boundary resets
    """
    if ssm_module is None:
        d_model = x.shape[-1]
        ssm_module = MambaVarlenScan(d_model).to(x.device)

    y, _ = ssm_module(x, sample_indices, use_reference=use_reference)
    return y
