import torch

from aeterna.graphs.sparse_graph import build_local_window_graph
from aeterna.graphs.laplacian_ops import normalized_laplacian_matvec
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


def main() -> None:
    torch.manual_seed(0)
    n = 64
    x = torch.randn(n, 16, requires_grad=True)
    adj = build_local_window_graph(x.detach(), window=4)
    est = vnge_hutchinson(adj, num_probes=16, cheb_order=12)
    exact = exact_vnge(adj)
    mae = torch.abs(est - exact)
    print(f"VNGE exact={exact.item():.4f} est={est.item():.4f} mae={mae.item():.4f}")
    # gradient check
    adj = build_local_window_graph(x, window=4)
    loss = vnge_hutchinson(adj, num_probes=8, cheb_order=8)
    loss.backward()
    grad_finite = torch.isfinite(x.grad).all().item()
    print(f"Gradient finite: {grad_finite}")


if __name__ == "__main__":
    main()
