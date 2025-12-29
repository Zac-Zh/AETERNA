import torch
import torch.nn.functional as F


def build_local_window_graph(x: torch.Tensor, window: int = 8, kernel: str = "cosine") -> torch.Tensor:
    """Build sparse adjacency for a sequence with local window edges.

    Args:
        x: (N, D) token representations
        window: local window radius
        kernel: cosine or gaussian
    Returns:
        sparse adjacency (N, N) as torch.sparse_coo_tensor
    """
    n, _ = x.shape
    if n == 0:
        return torch.sparse_coo_tensor(size=(0, 0), device=x.device)

    row_indices = []
    col_indices = []
    values = []
    for i in range(n):
        start = max(0, i - window)
        end = min(n, i + window + 1)
        neighbors = [j for j in range(start, end) if j != i]
        if not neighbors:
            continue
        xi = x[i].unsqueeze(0)
        xj = x[neighbors]
        if kernel == "cosine":
            sims = F.cosine_similarity(xi, xj, dim=-1)
            weights = (sims + 1.0) / 2.0
        elif kernel == "gaussian":
            dists = torch.cdist(xi, xj).squeeze(0)
            weights = torch.exp(-dists)
        else:
            raise ValueError(f"Unknown kernel: {kernel}")
        row_indices.extend([i] * len(neighbors))
        col_indices.extend(neighbors)
        values.append(weights)

    if not values:
        return torch.sparse_coo_tensor(size=(n, n), device=x.device)

    values = torch.cat(values)
    indices = torch.tensor([row_indices, col_indices], device=x.device)
    return torch.sparse_coo_tensor(indices, values, size=(n, n)).coalesce()
