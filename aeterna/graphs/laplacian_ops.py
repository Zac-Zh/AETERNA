import torch


def normalized_laplacian_matvec(adj: torch.Tensor, vec: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Compute L v where L = I - D^{-1/2} A D^{-1/2}.

    Args:
        adj: sparse adjacency (N, N)
        vec: (N,)
    """
    n = adj.size(0)
    if n == 0:
        return vec
    if not adj.is_sparse:
        raise ValueError("adjacency must be sparse")
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.rsqrt(deg + eps)
    scaled = deg_inv_sqrt * vec
    av = torch.sparse.mm(adj, scaled.unsqueeze(-1)).squeeze(-1)
    return vec - deg_inv_sqrt * av


def laplacian_trace(adj: torch.Tensor) -> torch.Tensor:
    """Compute trace of normalized Laplacian."""
    n = adj.size(0)
    if n == 0:
        return torch.tensor(0.0, device=adj.device)
    deg = torch.sparse.sum(adj, dim=1).to_dense()
    deg_inv_sqrt = torch.rsqrt(deg + 1e-8)
    diag = 1.0 - deg_inv_sqrt * deg_inv_sqrt * deg
    return diag.sum()
