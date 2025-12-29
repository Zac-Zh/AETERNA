import torch


def reset_state_on_boundaries(state: torch.Tensor, sample_indices: torch.Tensor) -> torch.Tensor:
    """Return a mask indicating where state should be reset.

    Args:
        state: (...)
        sample_indices: (T,)
    Returns:
        reset_mask: (T,) bool where True means reset before token t.
    """
    if sample_indices.numel() == 0:
        return torch.zeros(0, device=sample_indices.device, dtype=torch.bool)
    reset = torch.zeros_like(sample_indices, dtype=torch.bool)
    reset[0] = True
    reset[1:] = sample_indices[1:] != sample_indices[:-1]
    return reset
