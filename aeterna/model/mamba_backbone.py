import torch
from torch import nn


class SimpleRecurrentBlock(nn.Module):
    """A tiny stateful backbone compatible with packed resets."""

    def __init__(self, d_model: int):
        super().__init__()
        self.in_proj = nn.Linear(d_model, d_model)
        self.state_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.act = nn.Tanh()

    def step(self, x_t: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        h = self.act(self.in_proj(x_t) + self.state_proj(state))
        return self.out_proj(h), h

    def forward_packed(
        self, x: torch.Tensor, sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """Process packed tokens with boundary resets.

        Args:
            x: (T, D)
            sample_indices: (T,)
        Returns:
            outputs: (T, D)
        """
        device = x.device
        T, D = x.shape
        outputs = torch.zeros(T, D, device=device, dtype=x.dtype)
        state = torch.zeros(D, device=device, dtype=x.dtype)
        for t in range(T):
            if t > 0 and sample_indices[t] != sample_indices[t - 1]:
                state = torch.zeros_like(state)
            outputs[t], state = self.step(x[t], state)
        return outputs

    def forward_padded(self, x: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Process padded batches.

        Args:
            x: (B, L, D)
            attention_mask: (B, L)
        Returns:
            outputs: (B, L, D)
        """
        B, L, D = x.shape
        outputs = torch.zeros_like(x)
        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        for t in range(L):
            mask = attention_mask[:, t].unsqueeze(-1)
            step_out = self.act(self.in_proj(x[:, t]) + self.state_proj(state))
            state = self.out_proj(step_out)
            outputs[:, t] = state
            state = state * mask
        return outputs
