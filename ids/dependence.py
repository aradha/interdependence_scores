try:
    import torch
    backend = "torch"
except ImportError:
    backend = "numpy"

from .numpy_dependence import compute_IDS_numpy

if backend == "torch":
    from .torch_dependence import compute_IDS_torch

def compute_IDS(X, Y=None, num_terms=6, p_norm='max', p_val=False, num_tests=100):
    """Compute IDS between all pairs of variables in X (or between X and Y).

    Parameters:
        X: np.ndarray or torch.Tensor
        Y: np.ndarray or torch.Tensor (optional)
        num_terms: Number of terms for Taylor series approximation (optional)
        p_norm: String 'max' if using IDS-max.  1 or 2 for IDS-1, IDS-2, respectively.  (optional)
        p_val: Boolean.  Indicates whether to compute p-values using permutation tests
        num_tests: Number of permutation tests if p_val=True
    Returns:
        IDS matrix, p-value matrix (if p_val=True)
    """

    if backend == "numpy":
        return compute_IDS_numpy(X, Y=Y, num_terms=num_terms, p_norm=p_norm, p_val=p_val, num_tests=num_tests)
    elif backend == "torch":
        return compute_IDS_torch(X, Y=Y, num_terms=num_terms, p_norm=p_norm, p_val=p_val, num_tests=num_tests)
    else:
        raise ValueError("Unsupported backend: choose 'numpy' or 'torch'.")
