import torch

from aeterna.graphs.sparse_graph import build_local_window_graph
from aeterna.loss.vnge_loss import vnge_hutchinson


def exact_vnge(adj: torch.Tensor) -> torch.Tensor:
    dense = adj.to_dense()
    deg = dense.sum(dim=1)
    deg_inv_sqrt = torch.rsqrt(deg + 1e-8)
    d_inv_sqrt = torch.diag(deg_inv_sqrt)
    lap = torch.eye(adj.size(0), device=adj.device) - d_inv_sqrt @ dense @ d_inv_sqrt
    trace_l = torch.trace(lap)
    rho = lap / (trace_l + 1e-8)
    eigvals = torch.linalg.eigvalsh(rho)
    eigvals = torch.clamp(eigvals, min=1e-8)
    return -(eigvals * torch.log(eigvals)).sum()


def test_vnge_exact_small():
    torch.manual_seed(0)
    n = 32
    x = torch.randn(n, 8)
    adj = build_local_window_graph(x, window=3)
    est = vnge_hutchinson(adj, num_probes=64, cheb_order=16)
    exact = exact_vnge(adj)
    mae = torch.abs(est - exact)
    assert mae < 1e-2
