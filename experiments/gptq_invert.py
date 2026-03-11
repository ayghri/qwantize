
import numpy as np

def quantize_mxfp4_np(w_col, block_size=32, expo_offset=0):
    """
        Translates the provided PyTorch MXFP4 quantization to NumPy.
            w_col: Input vector (e.g. one column of weights).
    """
    # Define the lookup table
    doubled_mxfp4 = np.array([0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12])
    
    # Flatten and Pad
    orig_shape = w_col.shape
    w_flat = w_col.flatten()
    
    # Handle padding for block size
    remainder = w_flat.size % block_size
    if remainder != 0:
        pad_len = block_size - remainder
        w_padded = np.concatenate([w_flat, np.zeros(pad_len)])
    else:
        w_padded = w_flat
            
    num_blocks = w_padded.size // block_size
    w_reshaped = w_padded.reshape(num_blocks, block_size)
    
    # --- Scale Calculation ---
    # PyTorch: batch.view(...).abs().amax(dim=-1).log2().add(-2 + 127 + offset).floor()
    # scale = 2^(exponent - 128)
    
    # Avoid log2(0) by adding epsilon or masking
    abs_max = np.max(np.abs(w_reshaped), axis=-1)
    abs_max[abs_max == 0] = 1e-9 # Prevent log warning
    
    log2_max = np.log2(abs_max)
    scale_exponent = np.floor(log2_max - 2 + 127 + expo_offset)
    scale = np.power(2.0, scale_exponent - 128) # Shape: (num_blocks,)
    
    # --- Quantization ---
    # possible_values = scale.unsqueeze * table.unsqueeze
    # Shape: (num_blocks, 1, 16)
    possible_values = scale[:, np.newaxis, np.newaxis] * doubled_mxfp4[np.newaxis, np.newaxis, :]
    
    # w_reshaped: (num_blocks, block_size) -> (num_blocks, block_size, 1)
    w_expanded = w_reshaped[:, :, np.newaxis]
    
    # Deltas: |w - possible|
    # Shape: (num_blocks, block_size, 16)
    deltas = np.abs(w_expanded - possible_values)
    
    # Argmin
    indices = np.argmin(deltas, axis=-1)
    
    # Dequantize
    # doubled_fp4_quants = doubled_mxfp4[indices]
    quant_values = doubled_mxfp4[indices] # Shape: (num_blocks, block_size)
    dequants = scale[:, np.newaxis] * quant_values
    
    # Reshape back and crop padding
    dequants_flat = dequants.flatten()
    w_out = dequants_flat[:w_flat.size].reshape(orig_shape)
    
    return w_out

def run_experiment_mxfp4(d_row=128, d_col=128, strategy='sequential'):
    np.random.seed(42)
    
    # Setup Data
    W = np.random.randn(d_row, d_col)*0.1 +0.05
    X = np.random.randn(d_col, 512)

    
    # Hessian
    H = 2 * (X @ X.T)
    H = np.diag(np.diag(H)) + 0.1*(H*0+1.)
    damp = 0.01 * np.mean(np.diag(H))
    H += damp * np.eye(d_col)
    H_inv = np.linalg.inv(H)
    
    W_algo = W.copy()
    H_inv_algo = H_inv.copy()
    
    remaining_indices = list(range(d_col))
    
    # Pre-compute diagonal inverse for stability check
    # In full GPTQ, Cholesky is used. Here we use basic inversion update.
    
    for step in range(d_col):
        q_idx_in_remaining = 0
        
        if strategy == 'sequential':
            q_idx_in_remaining = 0 
            # q_idx_in_remaining = len(remaining_indices)//2
        else:
            # Score candidates
            best_score = float('inf') if strategy == 'greedy_min' else float('-inf')
            best_idx = -1
                
            for idx, col_idx in enumerate(remaining_indices):
                w_col = W_algo[:, col_idx]
                w_quant = quantize_mxfp4_np(w_col)
                
                # OBQ Error Metric: sum((w-q)^2) / [H^-1]_qq
                err_num = np.sum((w_col - w_quant) ** 2)
                err_denom = H_inv_algo[col_idx, col_idx]
                
                if err_denom <= 1e-9: err_denom = 1e-9
                    
                error = err_num / err_denom
                
                if strategy == 'greedy_min':
                    if error < best_score:
                        best_score, best_idx = error, idx
                elif strategy == 'greedy_max':
                    if error > best_score:
                        best_score, best_idx = error, idx
                    
            q_idx_in_remaining = best_idx

        q = remaining_indices[q_idx_in_remaining]
        
        # Quantize Selected
        w_col = W_algo[:, q]
        w_quant = quantize_mxfp4_np(w_col)
        diff = w_col - w_quant
        W_algo[:, q] = w_quant 
        
        # Update Weights
        h_qq = H_inv_algo[q, q]
        if abs(h_qq) < 1e-9: h_qq = 1e-9
            
        h_row = H_inv_algo[q, :]
        update_term = (diff / h_qq)[:, None] @ h_row[None, :]
        
        mask = np.zeros(d_col, dtype=bool)
        mask[remaining_indices] = True
        mask[q] = False
        
        W_algo[:, mask] -= update_term[:, mask]
        
        # Update Hessian Inverse
        row_col_term = h_row[:, None] @ h_row[None, :]
        H_inv_algo -= row_col_term / h_qq
        
        remaining_indices.pop(q_idx_in_remaining)

        # Final Evaluation
    error_fro = np.linalg.norm(W @ X - W_algo @ X) ** 2
    return error_fro

# Run
loss_seq = run_experiment_mxfp4(strategy='sequential')
loss_min = run_experiment_mxfp4(strategy='greedy_min')
loss_max = run_experiment_mxfp4(strategy='greedy_max')

print(f"Sequential: {loss_seq:.4f}")
print(f"Greedy Min: {loss_min:.4f}")
print(f"Greedy Max: {loss_max:.4f}")
