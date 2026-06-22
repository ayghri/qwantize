"""Batched GPU SPGL1 LASSO solver in PyTorch.

Solves, for each row i in {0, ..., B-1}, the LASSO problem

    min_{x_i in R^n}  ||A x_i - b_i||_2   s.t.   ||x_i||_1 <= tau_i

where the operator A is shared across rows (or supplied as a generic
matvec/rmatvec pair). The Spectral Projected Gradient with non-monotone
backtracking line search and Barzilai-Borwein step is implemented per
row, so rows progress independently within a single batched kernel call.

Mirrors the reference https://github.com/drrelyea/spgl1 (LASSO mode only,
i.e. ``tau`` given, ``sigma=0``). Differences from the reference:
  - Batched: B independent problems share matvec/rmatvec calls
  - Per-row step size, per-row line-search progress, per-row history
  - No subspace minimization (LSQR) — not needed for moderately sized n

Reference for the algorithm:
  E. van den Berg and M. P. Friedlander, "Probing the Pareto frontier for
  basis pursuit solutions", SIAM J. Sci. Comput., 31(2):890-912, 2008.
"""

import torch


# ---------------------------------------------------------------------------
# Projection onto L1 ball
# ---------------------------------------------------------------------------


def project_l1_ball_batched(x, tau):
    """Project each row of *x* onto its L1 ball of radius *tau*.

    For each row independently solves
        min_{y}  ||y - x||_2^2   s.t.   ||y||_1 <= tau

    Algorithm: Duchi et al. (2008) — sort |x| descending, find the active
    set size rho via the cumulative-sum criterion, soft-threshold by
    theta = (cumsum_{<=rho} |x| - tau) / rho. Vectorized over the batch.

    Args:
        x: (B, N) tensor.
        tau: scalar (broadcast) or (B,) per-row L1 budget. Non-negative.

    Returns:
        (B, N) projection. For rows where ||x||_1 <= tau already, the
        original row is returned unchanged.
    """
    if x.ndim != 2:
        raise ValueError(f"Expected 2-D input, got shape {tuple(x.shape)}.")
    B, N = x.shape

    if not torch.is_tensor(tau):
        tau = torch.full((B,), float(tau), device=x.device, dtype=x.dtype)
    elif tau.ndim == 0:
        tau = tau.expand(B).to(device=x.device, dtype=x.dtype)
    else:
        tau = tau.to(device=x.device, dtype=x.dtype)

    sign = x.sign()
    a = x.abs()

    cur_l1 = a.sum(dim=-1)  # (B,)
    needs_proj = cur_l1 > tau  # (B,)

    sorted_a, _ = a.sort(dim=-1, descending=True)  # (B, N)
    cs = sorted_a.cumsum(dim=-1)  # (B, N)
    k_range = torch.arange(1, N + 1, device=x.device, dtype=x.dtype).unsqueeze(0)
    alphas = (cs - tau.unsqueeze(-1)) / k_range  # (B, N)

    # rho = number of active entries = largest k where sorted_a[k] > alphas[k]
    active = sorted_a > alphas
    rho = active.sum(dim=-1).clamp(min=1)  # (B,)
    theta = alphas.gather(dim=-1, index=(rho - 1).unsqueeze(-1)).squeeze(-1)
    theta = theta.clamp(min=0)  # (B,)

    proj_abs = (a - theta.unsqueeze(-1)).clamp(min=0)
    proj = sign * proj_abs

    return torch.where(needs_proj.unsqueeze(-1), proj, x)


def l1_norm_batched(x):
    """Per-row L1 norm of (B, N) -> (B,)."""
    return x.abs().sum(dim=-1)


def linf_norm_batched(x):
    """Per-row L-inf norm of (B, N) -> (B,)."""
    return x.abs().max(dim=-1).values


# ---------------------------------------------------------------------------
# Default operator for a shared dense A
# ---------------------------------------------------------------------------


def make_dense_op(A):
    """Build (matvec, rmatvec) closures for a dense operator A : R^n -> R^m.

    Both functions act batched: given (B, n) they return (B, m), and
    vice versa.

    Args:
        A: (m, n) tensor.

    Returns:
        ``(matvec, rmatvec)`` with signatures
        ``matvec((B, n)) -> (B, m)`` and ``rmatvec((B, m)) -> (B, n)``.
    """
    AT = A.T.contiguous()

    def matvec(x):  # (B, n) -> (B, m)
        return x @ AT

    def rmatvec(r):  # (B, m) -> (B, n)
        return r @ A

    return matvec, rmatvec


# ---------------------------------------------------------------------------
# SPGL1 LASSO main loop (batched)
# ---------------------------------------------------------------------------

# Exit codes
EXIT_OPTIMAL = 1
EXIT_ITERATIONS = 2
EXIT_LINE_FAIL = 3


def spgl1_lasso_batched(
    matvec,
    rmatvec,
    b,
    tau,
    n,
    x0=None,
    max_iter=200,
    n_prev_vals=3,
    opt_tol=1e-4,
    step_min=1e-16,
    step_max=1e5,
    max_line_iters=10,
    gamma=1e-4,
    verbose=False,
):
    """Batched SPGL1 LASSO solver.

    Solves, for each row i independently,

        min_{x_i in R^n}  0.5 * ||A x_i - b_i||_2^2   s.t.   ||x_i||_1 <= tau

    Args:
        matvec: callable ``(B, n) -> (B, m)`` applying A row-by-row.
        rmatvec: callable ``(B, m) -> (B, n)`` applying A^T row-by-row.
        b: (B, m) right-hand sides.
        tau: scalar or (B,) per-row L1 budget. Must be > 0 here (LASSO).
        n: dimension of x.
        x0: optional (B, n) initial iterate (default: zeros).
        max_iter: outer iteration cap.
        n_prev_vals: non-monotone line-search history depth.
        opt_tol: optimality tolerance on the relative duality gap.
        step_{min,max}: clamp on the Barzilai-Borwein step.
        max_line_iters: line-search backtrack cap per outer iter.
        gamma: line-search sufficient-descent constant.
        verbose: print per-iter diagnostics if True.

    Returns:
        ``(x, r, info)`` where
        - ``x``: (B, n) final iterate (best-objective restore).
        - ``r``: (B, m) final residual ``b - A x``.
        - ``info``: dict with diagnostic time series and counters.
    """
    B, m = b.shape
    device = b.device
    dtype = b.dtype

    if x0 is None:
        x = torch.zeros(B, n, device=device, dtype=dtype)
    else:
        x = x0.clone()

    # Project to the L1 ball
    x = project_l1_ball_batched(x, tau)

    # Initial residual, gradient
    r = b - matvec(x)
    g = -rmatvec(r)
    f = 0.5 * (r * r).sum(dim=-1)  # (B,)

    # Best so far
    fbest = f.clone()
    xbest = x.clone()
    rbest = r.clone()

    # Non-monotone history
    last_fv = torch.full((B, n_prev_vals), -float("inf"), device=device, dtype=dtype)
    last_fv[:, 0] = f

    # Initial BB step from projected gradient direction
    dx = project_l1_ball_batched(x - g, tau) - x  # (B, n)
    dx_inf = linf_norm_batched(dx).clamp(min=1e-30)
    gstep = (1.0 / dx_inf).clamp(min=step_min, max=step_max)  # (B,)

    bnorm = b.norm(dim=-1)  # (B,)

    info = {
        "rnorm_hist": [],
        "xnorm1_hist": [],
        "gnorm_hist": [],
        "n_line_iters": 0,
        "n_matvec": 1,
        "n_rmatvec": 1,
        "exit_iter": None,
        "exit_stat": None,
    }

    for it in range(max_iter):
        gnorm = linf_norm_batched(-g)  # dual L1 norm = Linf
        rnorm = r.norm(dim=-1)  # (B,)
        # Relative duality gap (per row); see van den Berg / Friedlander.
        gap = (r * (r - b)).sum(dim=-1) + tau_per_row(tau, B, device, dtype) * gnorm
        rgap = gap.abs() / f.clamp(min=1.0)

        info["rnorm_hist"].append(rnorm.detach().cpu())
        info["xnorm1_hist"].append(l1_norm_batched(x).detach().cpu())
        info["gnorm_hist"].append(gnorm.detach().cpu())

        # Per-row optimality: relative gap small OR residual << ||b||
        opt = (rgap <= opt_tol) | (rnorm < opt_tol * bnorm)
        if verbose:
            print(
                f"  [it {it:4d}]  rnorm: max={rnorm.max():.4e}  "
                f"med={rnorm.median():.4e}  "
                f"rgap: max={rgap.max():.2e}  "
                f"opt: {opt.sum().item()}/{B}",
                flush=True,
            )
        if opt.all():
            info["exit_stat"] = EXIT_OPTIMAL
            info["exit_iter"] = it
            break

        # ---- Projected backtracking line search (per-row step) ----
        step = torch.ones(B, device=device, dtype=dtype)
        line_done = torch.zeros(B, dtype=torch.bool, device=device)
        fmax = last_fv.max(dim=-1).values  # (B,)

        x_keep = x.clone()
        r_keep = r.clone()
        f_keep = f.clone()

        xold = x.clone()
        gold = g.clone()
        fold = f.clone()
        rold = r.clone()

        for li in range(max_line_iters):
            # Trial direction: gstep * g, scaled per-row by `step`
            step_g = (step * gstep).unsqueeze(-1)  # (B, 1)
            xnew = project_l1_ball_batched(x - step_g * g, tau)
            rnew = b - matvec(xnew)
            fnew = 0.5 * (rnew * rnew).sum(dim=-1)
            s = xnew - x
            gts = step * (g * s).sum(dim=-1)  # (B,)

            descent = fnew < fmax + gamma * step * gts
            nodescent = gts >= 0
            just_done = descent & ~line_done

            # Save successful step for rows that just converged
            if just_done.any():
                x_keep = torch.where(just_done.unsqueeze(-1), xnew, x_keep)
                r_keep = torch.where(just_done.unsqueeze(-1), rnew, r_keep)
                f_keep = torch.where(just_done, fnew, f_keep)

            line_done = line_done | descent | nodescent
            info["n_matvec"] += 1
            info["n_line_iters"] += 1
            if line_done.all():
                break
            # halve step on rows still searching
            step = torch.where(line_done, step, step * 0.5)

        # Rows that never satisfied descent — accept last try (no error fatal)
        if (~line_done).any():
            not_done = ~line_done
            x_keep = torch.where(not_done.unsqueeze(-1), xnew, x_keep)
            r_keep = torch.where(not_done.unsqueeze(-1), rnew, r_keep)
            f_keep = torch.where(not_done, fnew, f_keep)

        x = x_keep
        r = r_keep
        f = f_keep

        # ---- Gradient update + Barzilai-Borwein step ----
        g_new = -rmatvec(r)
        info["n_rmatvec"] += 1

        s_step = x - xold
        y_step = g_new - gold
        sts = (s_step * s_step).sum(dim=-1)
        sty = (s_step * y_step).sum(dim=-1)
        gstep_new = torch.where(
            sty > 0,
            (sts / sty.clamp(min=1e-30)).clamp(min=step_min, max=step_max),
            torch.full_like(sts, step_max),
        )
        gstep = gstep_new
        g = g_new

        # Update best
        better = f < fbest
        if better.any():
            xbest = torch.where(better.unsqueeze(-1), x, xbest)
            rbest = torch.where(better.unsqueeze(-1), r, rbest)
            fbest = torch.where(better, f, fbest)

        # Update non-monotone history
        last_fv[:, it % n_prev_vals] = f

    else:
        info["exit_stat"] = EXIT_ITERATIONS
        info["exit_iter"] = max_iter

    # Restore best
    x = xbest
    r = rbest
    info["fbest"] = fbest.detach().cpu()
    info["xnorm1_final"] = l1_norm_batched(x).detach().cpu()
    info["rnorm_final"] = r.norm(dim=-1).detach().cpu()

    return x, r, info


def tau_per_row(tau, B, device, dtype):
    """Return tau broadcast to shape (B,) tensor."""
    if torch.is_tensor(tau):
        if tau.ndim == 0:
            return tau.expand(B).to(device=device, dtype=dtype)
        return tau.to(device=device, dtype=dtype)
    return torch.full((B,), float(tau), device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Reduced (Gram-form) LASSO solver  — when A^T A is small and explicit
# ---------------------------------------------------------------------------


def spgl1_lasso_reduced_batched(
    H,
    ATb,
    b_norm_sq,
    tau,
    x0=None,
    max_iter=200,
    n_prev_vals=3,
    opt_tol=1e-4,
    step_min=1e-16,
    step_max=1e5,
    max_line_iters=10,
    gamma=1e-4,
    matvec_dtype=None,
    verbose=False,
):
    """Batched SPGL1 LASSO in **Gram form**.

    Solves, for each row i,

        min_{x_i in R^n}  0.5 * ||A x_i - b_i||_2^2   s.t.   ||x_i||_1 <= tau

    when ``H = A^T A`` (n x n) is known explicitly and small, so the
    algorithm can run entirely in n-dim space. The residual r is never
    materialized; this is the right form when ``m >> n`` (e.g., per-FP4-block
    quantization with n=16 and m=T_calibration).

    Required precomputed inputs (all on the same device, same dtype):
        H        : (n, n) Gram matrix A^T A. Symmetric PSD.
        ATb      : (B, n) projected RHS A^T b_i for each row.
        b_norm_sq: (B,)   ||b_i||_2^2 for each row (used only in f reporting).

    Returns ``(x, info)``. Residual norms in info are derived analytically.
    """
    if ATb.ndim != 2:
        raise ValueError(f"ATb must be (B, n), got {tuple(ATb.shape)}")
    B, n = ATb.shape
    device = ATb.device
    dtype = ATb.dtype

    # Optional low-precision matvec for the H matmul (dominant cost).
    # ATb, x, gradients stay in `dtype` (typically float32) — only the
    # H matmul casts down. bf16 keeps fp32 dynamic range with half memory.
    use_low_prec = matvec_dtype is not None and matvec_dtype != dtype
    if use_low_prec:
        H_mv = H.to(matvec_dtype).contiguous()
    else:
        H_mv = H

    def _Hx(x_in):
        if use_low_prec:
            return (x_in.to(matvec_dtype) @ H_mv).to(dtype)
        return x_in @ H_mv

    if x0 is None:
        x = torch.zeros(B, n, device=device, dtype=dtype)
    else:
        x = x0.clone()
    x = project_l1_ball_batched(x, tau)

    # f(x) = 0.5 (x^T H x - 2 <x, ATb> + ||b||^2)
    # g(x) = H x - ATb            (note: g_algo = -A^T r in original code)
    Hx = _Hx(x)
    f = 0.5 * ((x * Hx).sum(dim=-1) - 2.0 * (x * ATb).sum(dim=-1) + b_norm_sq)
    g = Hx - ATb

    fbest = f.clone()
    xbest = x.clone()

    last_fv = torch.full((B, n_prev_vals), -float("inf"), device=device, dtype=dtype)
    last_fv[:, 0] = f

    # Initial BB step from projected gradient direction
    dx = project_l1_ball_batched(x - g, tau) - x
    dx_inf = linf_norm_batched(dx).clamp(min=1e-30)
    gstep = (1.0 / dx_inf).clamp(min=step_min, max=step_max)

    bnorm = b_norm_sq.clamp(min=0).sqrt()

    info = {
        "n_line_iters": 0,
        "exit_iter": None,
        "exit_stat": None,
    }

    tau_t = tau_per_row(tau, B, device, dtype)

    for it in range(max_iter):
        gnorm = linf_norm_batched(-g)
        rnorm = (2.0 * f).clamp(min=0).sqrt()
        # gap = <r, r - b> + tau gnorm = ||r||^2 - <r, b> + tau gnorm
        # <r, b> = <Ax - b, b> = <x, A^T b> - ||b||^2 = <x, ATb> - ||b||^2
        rb = (x * ATb).sum(dim=-1) - b_norm_sq
        gap = (2.0 * f) - rb + tau_t * gnorm  # = ||r||^2 - <r,b> + tau gnorm
        rgap = gap.abs() / f.clamp(min=1.0)

        opt = (rgap <= opt_tol) | (rnorm < opt_tol * bnorm.clamp(min=1e-30))
        if verbose:
            print(
                f"  [r-it {it:4d}]  rnorm: max={rnorm.max():.4e}  "
                f"med={rnorm.median():.4e}  "
                f"opt: {opt.sum().item()}/{B}",
                flush=True,
            )
        if opt.all():
            info["exit_stat"] = EXIT_OPTIMAL
            info["exit_iter"] = it
            break

        # ---- Projected backtracking line search (per-row step) ----
        step = torch.ones(B, device=device, dtype=dtype)
        line_done = torch.zeros(B, dtype=torch.bool, device=device)
        fmax = last_fv.max(dim=-1).values

        x_keep = x.clone()
        f_keep = f.clone()

        xold = x.clone()
        gold = g.clone()
        fold = f.clone()

        xnew = x
        fnew = f

        for li in range(max_line_iters):
            step_g = (step * gstep).unsqueeze(-1)
            xnew = project_l1_ball_batched(x - step_g * g, tau)
            Hxnew = _Hx(xnew)
            fnew = 0.5 * (
                (xnew * Hxnew).sum(dim=-1) - 2.0 * (xnew * ATb).sum(dim=-1) + b_norm_sq
            )
            s = xnew - x
            gts = step * (g * s).sum(dim=-1)

            descent = fnew < fmax + gamma * step * gts
            nodescent = gts >= 0
            just_done = descent & ~line_done

            if just_done.any():
                x_keep = torch.where(just_done.unsqueeze(-1), xnew, x_keep)
                f_keep = torch.where(just_done, fnew, f_keep)

            line_done = line_done | descent | nodescent
            info["n_line_iters"] += 1
            if line_done.all():
                break
            step = torch.where(line_done, step, step * 0.5)

        if (~line_done).any():
            not_done = ~line_done
            x_keep = torch.where(not_done.unsqueeze(-1), xnew, x_keep)
            f_keep = torch.where(not_done, fnew, f_keep)

        x = x_keep
        f = f_keep

        # ---- Gradient + BB ----
        Hx = _Hx(x)
        g_new = Hx - ATb
        s_step = x - xold
        y_step = g_new - gold
        sts = (s_step * s_step).sum(dim=-1)
        sty = (s_step * y_step).sum(dim=-1)
        gstep = torch.where(
            sty > 0,
            (sts / sty.clamp(min=1e-30)).clamp(min=step_min, max=step_max),
            torch.full_like(sts, step_max),
        )
        g = g_new

        better = f < fbest
        if better.any():
            xbest = torch.where(better.unsqueeze(-1), x, xbest)
            fbest = torch.where(better, f, fbest)

        last_fv[:, it % n_prev_vals] = f
    else:
        info["exit_stat"] = EXIT_ITERATIONS
        info["exit_iter"] = max_iter

    return xbest, info
