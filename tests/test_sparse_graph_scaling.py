"""Test that sparse graph construction scales linearly, not quadratically."""
import time
import torch
import pytest

from aeterna.graphs.sparse_graph import build_local_window_graph


def test_local_window_graph_is_sparse():
    """Test that local window graph returns sparse tensor."""
    torch.manual_seed(42)
    n = 100
    d = 16
    x = torch.randn(n, d)

    adj = build_local_window_graph(x, window=8)

    assert adj.is_sparse, "Adjacency should be sparse COO tensor"
    assert adj.shape == (n, n), f"Expected shape ({n}, {n}), got {adj.shape}"


def test_local_window_edges_count():
    """Test that edge count is O(N * window), not O(N^2)."""
    torch.manual_seed(42)
    window = 8

    for n in [50, 100, 200]:
        x = torch.randn(n, 16)
        adj = build_local_window_graph(x, window=window)

        # Count edges
        num_edges = adj._nnz()

        # Expected edges per node: ~2*window (both directions)
        # Total expected: ~n * 2 * window
        expected_edges = n * 2 * window

        # Allow some slack for boundary nodes
        assert num_edges <= expected_edges * 1.2, \
            f"Too many edges: {num_edges} > {expected_edges * 1.2} for n={n}"

        # Should not be O(N^2)
        assert num_edges < n * n * 0.5, \
            f"Edge count {num_edges} too close to dense {n*n}"


def test_local_window_no_dense_adjacency():
    """Test that we never construct full dense N×N adjacency."""
    torch.manual_seed(42)
    n = 512
    d = 16
    window = 8

    x = torch.randn(n, d)

    # This should NOT create a dense (512, 512) matrix
    adj = build_local_window_graph(x, window=window)

    # Verify it's sparse
    assert adj.is_sparse, "Large graph must be sparse"

    # Verify edge count is small
    num_edges = adj._nnz()
    max_allowed_edges = n * window * 3  # Conservative bound

    assert num_edges < max_allowed_edges, \
        f"Edge count {num_edges} suggests dense construction"


def test_sparse_graph_scaling_time():
    """Test that construction time scales linearly with N."""
    window = 8
    d = 16
    times = []
    sizes = [100, 200, 400]

    for n in sizes:
        torch.manual_seed(42)
        x = torch.randn(n, d)

        start = time.perf_counter()
        adj = build_local_window_graph(x, window=window)
        elapsed = time.perf_counter() - start
        times.append(elapsed)

        # Force computation
        _ = adj._nnz()

    # Time should scale roughly linearly
    # time[1] / time[0] ≈ sizes[1] / sizes[0] = 2
    # time[2] / time[1] ≈ sizes[2] / sizes[1] = 2
    # If O(N^2), ratios would be ~4

    ratio_1 = times[1] / max(times[0], 1e-6)
    ratio_2 = times[2] / max(times[1], 1e-6)

    # Should be closer to 2 (linear) than 4 (quadratic)
    # Use lenient bounds since timing can be noisy
    assert ratio_1 < 3.5, \
        f"Scaling looks quadratic: ratio {ratio_1:.2f} (expected ~2 for linear)"
    assert ratio_2 < 3.5, \
        f"Scaling looks quadratic: ratio {ratio_2:.2f} (expected ~2 for linear)"


def test_sparse_graph_no_oom_large():
    """Test that we can build graphs for large sequences without OOM.

    This test ensures we don't accidentally construct dense O(N^2) matrices.
    """
    torch.manual_seed(42)
    # Use a size that would OOM if we created dense (2048, 2048) adjacency
    n = 2048
    d = 16
    window = 8

    x = torch.randn(n, d)

    # This should succeed without OOM
    try:
        adj = build_local_window_graph(x, window=window)
        assert adj.is_sparse, "Must be sparse for large N"
        assert adj._nnz() < n * window * 3, "Too many edges"
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            pytest.fail("OOM error suggests dense adjacency construction")
        else:
            raise


def test_zero_length_sequence():
    """Test edge case of zero-length sequence."""
    x = torch.randn(0, 16)
    adj = build_local_window_graph(x, window=8)

    assert adj.shape == (0, 0), "Empty input should give empty adjacency"
    assert adj._nnz() == 0, "Empty input should have no edges"
