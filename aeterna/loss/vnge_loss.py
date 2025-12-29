import math
from typing import Tuple

import torch

from aeterna.graphs.laplacian_ops import normalized_laplacian_matvec
from aeterna.graphs.sparse_graph import build_local_window_graph


def _chebyshev_coeffs(func, order: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Compute Chebyshev coefficients for func on [-1, 1]."""
    n = order + 1
    k = torch.arange(n, device=device, dtype=dtype)
    theta = math.pi * (k + 0.5) / n
    x = torch.cos(theta)
    fx = func(x)
    coeffs = torch.zeros(n, device=device, dtype=dtype)
    for j in range(n):
        coeffs[j] = (2.0 / n) * torch.sum(fx * torch.cos(j * theta))
    coeffs[0] *= 0.5
    return coeffs


def _apply_chebyshev(
    matvec,
    v: torch.Tensor,
    coeffs: torch.Tensor,
) -> torch.Tensor:
    """Apply Chebyshev polynomial approximation to vector."""
    if len(coeffs) == 1:
        return coeffs[0] * v
    t0 = v
    t1 = matvec(v)
    out = coeffs[0] * t0 + coeffs[1] * t1
    for k in range(2, len(coeffs)):
        t2 = 2.0 * matvec(t1) - t0
        out = out + coeffs[k] * t2
        t0, t1 = t1, t2
    return out


def vnge_hutchinson(
    adj: torch.Tensor,
    num_probes: int = 8,
    cheb_order: int = 12,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Estimate VNGE using Hutchinson + Chebyshev on sparse graphs."""
    n = adj.size(0)
    if n == 0:
        return torch.tensor(0.0, device=adj.device)
    if not adj.is_sparse:
        raise ValueError("Adjacency must be sparse")
    # Use direct computation as specified: trace_l = n for normalized Laplacian
    trace_l = torch.tensor(float(n), device=adj.device, dtype=adj.dtype) + eps

    def matvec_l(v):
        return normalized_laplacian_matvec(adj, v, eps=eps)

    def matvec_tilde(v):
        return matvec_l(v) - v

    def func(x):
        # x in [-1, 1], map to lambda in [0, 2]
        lam = x + 1.0
        rho = lam / trace_l
        rho = torch.clamp(rho, min=eps)
        return -(rho * torch.log(rho))

    coeffs = _chebyshev_coeffs(func, cheb_order, adj.device, trace_l.dtype)
    estimates = []
    for _ in range(num_probes):
        z = torch.randint(0, 2, (n,), device=adj.device, dtype=trace_l.dtype)
        z = z * 2 - 1
        fz = _apply_chebyshev(matvec_tilde, z, coeffs)
        estimates.append(torch.dot(z, fz))
    return torch.stack(estimates).mean()


def vnge_loss(
    x: torch.Tensor,
    sample_indices: torch.Tensor,
    window: int = 8,
    num_probes: int = 8,
    cheb_order: int = 12,
    eps: float = 1e-8,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute VNGE per sequence and return mean and per-seq values."""
    vnge_values = []
    if x.numel() == 0:
        return torch.tensor(0.0, device=x.device), torch.tensor([], device=x.device)
    unique_ids = torch.unique(sample_indices)
    for sid in unique_ids:
        mask = sample_indices == sid
        x_seq = x[mask]
        adj = build_local_window_graph(x_seq, window=window)
        vnge = vnge_hutchinson(adj, num_probes=num_probes, cheb_order=cheb_order, eps=eps)
        vnge_values.append(vnge)
    vnge_tensor = torch.stack(vnge_values)
    return vnge_tensor.mean(), vnge_tensor
