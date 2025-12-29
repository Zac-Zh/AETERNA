import torch
from torch import nn
import torch.nn.functional as F

from aeterna.model.mamba_backbone import SimpleRecurrentBlock


class AeternaLM(nn.Module):
    def __init__(self, vocab_size: int, d_model: int = 128):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.backbone = SimpleRecurrentBlock(d_model)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size)

    def forward_packed(self, input_ids: torch.Tensor, sample_indices: torch.Tensor):
        x = self.embed(input_ids)
        hidden = self.backbone.forward_packed(x, sample_indices)
        hidden = self.ln(hidden)
        logits = self.head(hidden)
        return logits, hidden

    def forward_padded(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        x = self.embed(input_ids)
        hidden = self.backbone.forward_padded(x, attention_mask)
        hidden = self.ln(hidden)
        logits = self.head(hidden)
        return logits, hidden

    def loss_packed(self, input_ids: torch.Tensor, labels: torch.Tensor, sample_indices: torch.Tensor):
        logits, hidden = self.forward_packed(input_ids, sample_indices)
        loss = F.cross_entropy(logits, labels)
        return loss, logits, hidden

    def loss_padded(self, input_ids: torch.Tensor, labels: torch.Tensor, attention_mask: torch.Tensor):
        logits, hidden = self.forward_padded(input_ids, attention_mask)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
        return loss, logits, hidden
