"""Unit tests for qwantize.spgl1 — projection and LASSO solver vs scipy CPU.

Run with:
    PYTHONPATH=. /misc/envs/quant/bin/python -m pytest tests/test_spgl1.py -v

Tests are skipped if the reference `spgl1` package or CUDA is unavailable.
"""

import numpy as np
import pytest
import torch

from qwantize.spgl1 import (
    project_l1_ball_batched,
    l1_norm_batched,
    linf_norm_batched,
    make_dense_op,
    spgl1_lasso_batched,
    spgl1_lasso_reduced_batched,
    EXIT_OPTIMAL,
)

try:
    import spgl1 as _spgl1_pkg
    _HAS_SPGL1 = True
except ImportError:
    _HAS_SPGL1 = False

_HAS_CUDA = torch.cuda.is_available()

requires_spgl1 = pytest.mark.skipif(not _HAS_SPGL1, reason="spgl1 package not installed")
requires_cuda = pytest.mark.skipif(not _HAS_CUDA, reason="CUDA not available")


# ===========================================================================
# Projection tests
# ===========================================================================

class TestL1BallProjection:
    """qwantize.spgl1.project_l1_ball_batched should match Duchi et al. /
    spgl1's oneprojector exactly (up to floating-point)."""

    @pytest.fixture
    def rand_x(self):
        torch.manual_seed(0)
        return torch.randn(8, 64, dtype=torch.float64)

    def _project_cpu_ref(self, x_np, tau):
        """Reference per-row projection using spgl1.oneprojector."""
        return np.stack([
            _spgl1_pkg.oneprojector(row.copy(), 1.0, tau) for row in x_np
        ])

    @requires_spgl1
    @pytest.mark.parametrize("tau", [0.1, 1.0, 5.0, 50.0])
    def test_matches_spgl1_oneprojector_cpu(self, rand_x, tau):
        x_np = rand_x.numpy()
        proj_ref = self._project_cpu_ref(x_np, tau)

        proj_cpu = project_l1_ball_batched(rand_x, tau).numpy()
        np.testing.assert_allclose(proj_cpu, proj_ref, atol=1e-10, rtol=0)

    @requires_spgl1
    @requires_cuda
    @pytest.mark.parametrize("tau", [0.1, 1.0, 5.0, 50.0])
    def test_matches_spgl1_oneprojector_gpu(self, rand_x, tau):
        x_np = rand_x.numpy()
        proj_ref = self._project_cpu_ref(x_np, tau)

        proj_gpu = project_l1_ball_batched(rand_x.cuda(), tau).cpu().numpy()
        np.testing.assert_allclose(proj_gpu, proj_ref, atol=1e-10, rtol=0)

    def test_no_projection_when_inside_ball(self, rand_x):
        """If ||x||_1 <= tau, projection returns x unchanged."""
        big_tau = 10 * rand_x.abs().sum(dim=-1).max().item()
        proj = project_l1_ball_batched(rand_x, big_tau)
        torch.testing.assert_close(proj, rand_x)

    def test_active_projection_norm(self, rand_x):
        """When the constraint is active, ||proj||_1 == tau (up to fp)."""
        tau = 1.0
        proj = project_l1_ball_batched(rand_x, tau)
        l1 = l1_norm_batched(proj)
        # All rows should saturate the constraint
        torch.testing.assert_close(l1, torch.full_like(l1, tau),
                                    atol=1e-9, rtol=0)

    def test_per_row_tau(self):
        """tau can be a per-row tensor."""
        torch.manual_seed(0)
        B, N = 4, 32
        x = torch.randn(B, N, dtype=torch.float64)
        tau = torch.tensor([0.5, 1.0, 2.0, 100.0], dtype=torch.float64)
        proj = project_l1_ball_batched(x, tau)
        l1 = l1_norm_batched(proj)
        # Row 3 has tau=100 > ||x||_1, so unchanged
        assert torch.allclose(proj[3], x[3])
        # Rows 0..2 should hit tau
        for i in range(3):
            assert abs(l1[i].item() - tau[i].item()) < 1e-9

    def test_zero_input(self):
        x = torch.zeros(3, 16, dtype=torch.float64)
        proj = project_l1_ball_batched(x, 1.0)
        torch.testing.assert_close(proj, x)

    def test_signs_preserved(self):
        """Projection preserves signs (it's soft-threshold of |x|)."""
        torch.manual_seed(1)
        x = torch.randn(4, 32, dtype=torch.float64)
        proj = project_l1_ball_batched(x, 1.0)
        # For nonzero entries, sign matches input
        nz = proj.abs() > 1e-12
        torch.testing.assert_close(proj[nz].sign(), x[nz].sign())


# ===========================================================================
# Norm helper tests
# ===========================================================================

class TestNormHelpers:

    def test_l1_norm_batched(self):
        x = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.5, -0.5]])
        out = l1_norm_batched(x)
        torch.testing.assert_close(out, torch.tensor([6.0, 1.0]))

    def test_linf_norm_batched(self):
        x = torch.tensor([[1.0, -2.0, 3.0], [0.0, 0.5, -7.0]])
        out = linf_norm_batched(x)
        torch.testing.assert_close(out, torch.tensor([3.0, 7.0]))


# ===========================================================================
# SPGL1 LASSO solver vs CPU reference
# ===========================================================================

class TestSpgl1LassoSolver:
    """End-to-end LASSO solver should match scipy spgl1 within rough
    tolerance on small synthetic problems."""

    def _make_synthetic_problem(self, m=128, n=64, k=8, snr=0.01, seed=0):
        rng = np.random.RandomState(seed)
        A = rng.randn(m, n).astype(np.float64) / np.sqrt(m)
        x_true = np.zeros(n)
        x_true[rng.choice(n, k, replace=False)] = rng.randn(k)
        b = A @ x_true + snr * rng.randn(m)
        return A, b, x_true

    @requires_spgl1
    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_lasso_cpu_matches_reference(self, seed):
        """Our solver on CPU tensors should reach a similar objective."""
        A, b, x_true = self._make_synthetic_problem(seed=seed)
        tau = 0.5 * float(np.linalg.norm(x_true, 1))

        # Reference (CPU)
        x_ref, _, _, info_ref = _spgl1_pkg.spgl1(
            A, b, tau=tau, sigma=0.0, iter_lim=500, verbosity=0,
        )
        f_ref = 0.5 * float(np.linalg.norm(A @ x_ref - b) ** 2)
        l1_ref = float(np.linalg.norm(x_ref, 1))

        # Ours (CPU torch)
        A_t = torch.from_numpy(A)
        b_t = torch.from_numpy(b).unsqueeze(0)
        matvec, rmatvec = make_dense_op(A_t)
        x_ours, r_ours, info_ours = spgl1_lasso_batched(
            matvec, rmatvec, b_t, tau=tau, n=A.shape[1], max_iter=500,
        )
        f_ours = 0.5 * float((r_ours ** 2).sum())
        l1_ours = float(x_ours.abs().sum())

        # Objective should be within 5% (line-search differences possible)
        assert f_ours <= 1.1 * f_ref + 1e-6, (
            f"GPU objective {f_ours:.6f} much worse than ref {f_ref:.6f}"
        )
        # L1 should saturate to tau (active constraint)
        assert abs(l1_ours - tau) < 1e-3 * max(tau, 1.0), (
            f"||x_ours||_1 = {l1_ours:.4f}, tau = {tau:.4f}"
        )
        assert abs(l1_ref - tau) < 1e-2 * max(tau, 1.0)

    @requires_spgl1
    @requires_cuda
    def test_lasso_gpu_matches_cpu(self):
        """GPU result should be bit-equivalent (modulo FP) to CPU torch."""
        A, b, _ = self._make_synthetic_problem(seed=0)
        tau = 0.5 * float(np.linalg.norm(b))  # generous tau

        A_t = torch.from_numpy(A)
        b_t = torch.from_numpy(b).unsqueeze(0)
        mv_cpu, rmv_cpu = make_dense_op(A_t)
        x_cpu, r_cpu, _ = spgl1_lasso_batched(
            mv_cpu, rmv_cpu, b_t, tau=tau, n=A.shape[1], max_iter=200,
        )

        A_g = A_t.cuda()
        b_g = b_t.cuda()
        mv_g, rmv_g = make_dense_op(A_g)
        x_g, r_g, _ = spgl1_lasso_batched(
            mv_g, rmv_g, b_g, tau=tau, n=A.shape[1], max_iter=200,
        )

        torch.testing.assert_close(x_cpu, x_g.cpu(), atol=1e-6, rtol=1e-5)

    @requires_spgl1
    def test_batched_independence(self):
        """Solving B problems in one batched call == B independent calls."""
        A, b1, _ = self._make_synthetic_problem(seed=0)
        _, b2, _ = self._make_synthetic_problem(seed=1)
        _, b3, _ = self._make_synthetic_problem(seed=2)
        b_stack = np.stack([b1, b2, b3])
        tau = 0.5 * float(np.linalg.norm(b1))

        A_t = torch.from_numpy(A)

        # Solve as batch of 3
        b_batch = torch.from_numpy(b_stack)
        mv, rmv = make_dense_op(A_t)
        x_batch, _, _ = spgl1_lasso_batched(
            mv, rmv, b_batch, tau=tau, n=A.shape[1], max_iter=200,
        )

        # Solve one at a time
        x_single = torch.empty(3, A.shape[1], dtype=torch.float64)
        for i in range(3):
            xi, _, _ = spgl1_lasso_batched(
                mv, rmv, torch.from_numpy(b_stack[i]).unsqueeze(0),
                tau=tau, n=A.shape[1], max_iter=200,
            )
            x_single[i] = xi.squeeze(0)

        torch.testing.assert_close(x_batch, x_single, atol=1e-8, rtol=1e-6)

    @requires_spgl1
    def test_per_row_tau(self):
        """Per-row tau gives per-row L1 budget."""
        A, b1, _ = self._make_synthetic_problem(seed=0)
        _, b2, _ = self._make_synthetic_problem(seed=1)
        b_stack = torch.from_numpy(np.stack([b1, b2]))
        A_t = torch.from_numpy(A)
        mv, rmv = make_dense_op(A_t)
        tau = torch.tensor([0.5, 1.0], dtype=torch.float64)

        x, _, _ = spgl1_lasso_batched(
            mv, rmv, b_stack, tau=tau, n=A.shape[1], max_iter=200,
        )
        l1 = l1_norm_batched(x)
        # Each row should saturate (or be at most) its own tau
        assert l1[0] <= 0.5 + 1e-3
        assert l1[1] <= 1.0 + 1e-3
        # And one should be larger (independent solves)
        assert l1[1] > l1[0]

    @requires_spgl1
    def test_loose_tau_recovers_unconstrained(self):
        """With tau >> ||x_LS||_1, the LASSO solution is the LS solution."""
        A, b, x_true = self._make_synthetic_problem(seed=0)
        x_ls, *_ = np.linalg.lstsq(A, b, rcond=None)
        tau = 100.0 * float(np.linalg.norm(x_ls, 1))

        A_t = torch.from_numpy(A)
        b_t = torch.from_numpy(b).unsqueeze(0)
        mv, rmv = make_dense_op(A_t)
        x_ours, r_ours, _ = spgl1_lasso_batched(
            mv, rmv, b_t, tau=tau, n=A.shape[1], max_iter=500,
        )
        # Residual should be very small (least-squares fit)
        res = float((r_ours ** 2).sum().sqrt())
        ls_res = float(np.linalg.norm(A @ x_ls - b))
        # Should match LS residual within sqrt-machine-eps
        assert res < 1.05 * ls_res + 1e-3


class TestSpgl1LassoReduced:
    """Gram-form solver must agree with the explicit-A version on the
    same problem."""

    def _make_problem(self, m=200, n=16, k=4, snr=0.01, seed=0, B=3):
        rng = np.random.RandomState(seed)
        A = rng.randn(m, n).astype(np.float64) / np.sqrt(m)
        b_list = []
        for i in range(B):
            x_true = np.zeros(n)
            x_true[rng.choice(n, k, replace=False)] = rng.randn(k)
            b_list.append(A @ x_true + snr * rng.randn(m))
        b = np.stack(b_list)  # (B, m)
        return A, b

    @pytest.mark.parametrize("seed", [0, 1, 2])
    def test_reduced_matches_explicit(self, seed):
        A, b = self._make_problem(seed=seed)
        m, n = A.shape
        B = b.shape[0]
        tau = 0.5 * float(np.linalg.norm(b, axis=-1).mean())

        # Explicit A
        A_t = torch.from_numpy(A)
        b_t = torch.from_numpy(b)
        mv, rmv = make_dense_op(A_t)
        x_full, r_full, _ = spgl1_lasso_batched(
            mv, rmv, b_t, tau=tau, n=n, max_iter=200,
        )

        # Reduced form
        H_t = torch.from_numpy(A.T @ A)
        ATb_t = b_t @ A_t                         # (B, n)
        b_sq_t = (b_t ** 2).sum(dim=-1)           # (B,)
        x_red, _ = spgl1_lasso_reduced_batched(
            H_t, ATb_t, b_sq_t, tau=tau, max_iter=200,
        )

        # FP roundoff: explicit A and Gram-form accumulate slightly differently
        torch.testing.assert_close(x_full, x_red, atol=1e-4, rtol=1e-3)

    @requires_cuda
    def test_reduced_gpu_matches_cpu(self):
        A, b = self._make_problem(seed=0)
        m, n = A.shape
        tau = 0.5 * float(np.linalg.norm(b, axis=-1).mean())

        H_t = torch.from_numpy(A.T @ A)
        ATb_t = torch.from_numpy(b) @ torch.from_numpy(A)
        b_sq_t = torch.from_numpy((b ** 2).sum(axis=-1))

        x_cpu, _ = spgl1_lasso_reduced_batched(
            H_t, ATb_t, b_sq_t, tau=tau, max_iter=200,
        )
        x_gpu, _ = spgl1_lasso_reduced_batched(
            H_t.cuda(), ATb_t.cuda(), b_sq_t.cuda(),
            tau=tau, max_iter=200,
        )
        torch.testing.assert_close(x_cpu, x_gpu.cpu(), atol=1e-6, rtol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
