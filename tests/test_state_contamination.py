import torch

from aeterna.model.aeterna_model import AeternaLM


def test_state_contamination():
    torch.manual_seed(0)
    vocab_size = 128
    model = AeternaLM(vocab_size=vocab_size, d_model=32)
    model.eval()

    seq_a = torch.randint(0, vocab_size, (10,))
    seq_b = torch.randint(0, vocab_size, (12,))

    packed = torch.cat([seq_a, seq_b], dim=0)
    sample_indices = torch.cat(
        [torch.zeros(len(seq_a), dtype=torch.long), torch.ones(len(seq_b), dtype=torch.long)],
        dim=0,
    )

    with torch.no_grad():
        logits_packed, _ = model.forward_packed(packed, sample_indices)
        logits_b_only, _ = model.forward_packed(seq_b, torch.zeros_like(seq_b))

    packed_b = logits_packed[len(seq_a) :]
    assert torch.allclose(packed_b, logits_b_only, atol=1e-5)
