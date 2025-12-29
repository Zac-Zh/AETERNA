import torch

from aeterna.graphs.sparse_graph import build_local_window_graph
from aeterna.loss.vnge_loss import vnge_hutchinson


def exact_vnge(adj: torch.Tensor) -> torch.Tensor:
    """Compute exact VNGE using eigendecomposition.

    Uses same trace approximation (trace_l = n) as vnge_hutchinson
    for consistent comparison.
    """
    n = adj.size(0)
    dense = adj.to_dense()
    deg = dense.sum(dim=1)
    deg_inv_sqrt = torch.rsqrt(deg + 1e-8)
    d_inv_sqrt = torch.diag(deg_inv_sqrt)
    lap = torch.eye(n, device=adj.device) - d_inv_sqrt @ dense @ d_inv_sqrt
    # Use same trace approximation as vnge_hutchinson: trace_l = n
    trace_l = torch.tensor(float(n), device=adj.device, dtype=adj.dtype)
    rho = lap / (trace_l + 1e-8)
    eigvals = torch.linalg.eigvalsh(rho)
    eigvals = torch.clamp(eigvals, min=1e-8)
    return -(eigvals * torch.log(eigvals)).sum()


def test_vnge_exact_small():
    """Test VNGE estimator accuracy against exact eigendecomposition.

    Note: Chebyshev polynomial approximation of f(x) = -x*log(x) combined with
    Hutchinson trace estimation introduces approximation error. An MAE of ~0.05
    is reasonable and expected for this stochastic estimator.
    """
    torch.manual_seed(0)
    n = 32
    x = torch.randn(n, 8)
    adj = build_local_window_graph(x, window=3)
    # Use sufficient probes and Chebyshev order for good accuracy
    est = vnge_hutchinson(adj, num_probes=128, cheb_order=24)
    exact = exact_vnge(adj)
    mae = torch.abs(est - exact)
    # Accept reasonable approximation error from Chebyshev + Hutchinson
    # This validates the estimator is working, not requiring perfect accuracy
    assert mae < 0.1, f"MAE {mae:.4f} too large (threshold 0.1)"
    print(f"VNGE estimator MAE: {mae:.4f} (exact={exact:.4f}, est={est:.4f})")
