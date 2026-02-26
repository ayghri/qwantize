import torch


def compute_metrics(W, W_dq, X=None):
    """Compute weight and output quantization error metrics.

    Args:
        W: Original weights of shape ``(M, K)``.
        W_dq: Dequantized weights of shape ``(M, K)``.
        X: Input activations of shape ``(T, K)``. If provided, output error
            metrics are also computed.

    Returns:
        Dict with the following keys:

        - ``"weight_error"``: ``||W_dq - W||_F``
        - ``"weight_error_pct"``: ``||W_dq - W||_F / ||W||_F * 100``
        - ``"output_error"``: ``||W_dq @ X.T - W @ X.T||_F`` (only if *X* given)
        - ``"output_error_pct"``: normalized output error in percent (only if *X* given)
    """
    metrics = {}

    # Weight error
    w_err = (W_dq.float() - W.float()).norm()
    w_norm = W.float().norm()
    metrics["weight_error"] = w_err.item()
    metrics["weight_error_pct"] = (w_err / w_norm * 100).item()

    if X is not None:
        # Output error: ||W_dq @ X.T - W @ X.T||_F
        # Batch over X rows to manage memory
        batch_size = 4096
        T = X.shape[0]
        W_f = W.float()
        W_dq_f = W_dq.float()
        sse_accum = 0.0
        ref_sse_accum = 0.0

        for i in range(0, T, batch_size):
            X_batch = X[i : i + batch_size].float()  # (batch, K)
            out_ref = X_batch @ W_f.T  # (batch, M)
            out_dq = X_batch @ W_dq_f.T  # (batch, M)
            diff = out_dq - out_ref
            sse_accum += diff.pow(2).sum().item()
            ref_sse_accum += out_ref.pow(2).sum().item()

        out_err = sse_accum**0.5
        out_norm = ref_sse_accum**0.5
        metrics["output_error"] = out_err
        metrics["output_error_pct"] = out_err / out_norm * 100

    return metrics
