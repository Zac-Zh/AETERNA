"""Test correctness of variable-length scan operations."""
import torch
import pytest

from aeterna.ops.varlen_scan import MambaVarlenScan


def test_varlen_scan_output_shape():
    """Test that varlen scan preserves shape."""
    torch.manual_seed(42)
    d_model = 64
    T = 100

    ssm = MambaVarlenScan(d_model, d_state=16)
    x = torch.randn(T, d_model)
    sample_indices = torch.randint(0, 5, (T,))

    y, _ = ssm(x, sample_indices, use_reference=True)

    assert y.shape == x.shape, f"Expected shape {x.shape}, got {y.shape}"


def test_varlen_scan_reference_vs_optimized():
    """Test that optimized path matches reference (when available)."""
    torch.manual_seed(42)
    d_model = 32
    T = 50

    ssm = MambaVarlenScan(d_model, d_state=8)
    ssm.eval()  # Disable dropout if any

    x = torch.randn(T, d_model)
    sample_indices = torch.randint(0, 3, (T,))

    with torch.no_grad():
        y_ref, _ = ssm(x, sample_indices, use_reference=True)
        y_opt, _ = ssm(x, sample_indices, use_reference=False)

    # Should match closely (may have small numerical differences)
    assert torch.allclose(y_ref, y_opt, atol=1e-5), \
        f"Reference and optimized outputs differ: max diff = {(y_ref - y_opt).abs().max()}"


def test_varlen_scan_gradients():
    """Test that gradients flow correctly through varlen scan."""
    torch.manual_seed(42)
    d_model = 32
    T = 30

    ssm = MambaVarlenScan(d_model, d_state=8)
    x = torch.randn(T, d_model, requires_grad=True)
    sample_indices = torch.randint(0, 2, (T,))

    y, _ = ssm(x, sample_indices, use_reference=True)
    loss = y.sum()
    loss.backward()

    assert x.grad is not None, "No gradient computed for input"
    assert torch.isfinite(x.grad).all(), "Non-finite gradients detected"
    assert (x.grad.abs() > 0).any(), "All gradients are zero"


def test_varlen_scan_deterministic():
    """Test that varlen scan is deterministic with same seed."""
    d_model = 32
    T = 40

    def run():
        torch.manual_seed(123)
        ssm = MambaVarlenScan(d_model, d_state=8)
        torch.manual_seed(123)
        x = torch.randn(T, d_model)
        sample_indices = torch.randint(0, 2, (T,))
        y, _ = ssm(x, sample_indices, use_reference=True)
        return y

    y1 = run()
    y2 = run()

    assert torch.allclose(y1, y2, atol=1e-6), \
        "Outputs differ with same seed (non-deterministic)"
