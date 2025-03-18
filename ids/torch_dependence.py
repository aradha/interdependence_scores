import torch 
import math
import sys
from tqdm import tqdm

SEED = 1717
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)

EPSILON = sys.float_info.epsilon

def transform(y, num_terms=6, bandwidth_term=1/2):
    B = bandwidth_term 
    exp = torch.exp(-B * y**2)
    terms = []
    for i in range(num_terms):
        terms.append(exp * (y)**i / math.sqrt(math.factorial(i) *1.))
    y_ = torch.cat(terms, axis=-1)
    return y_

def center(X):
    return X - torch.mean(X, axis=0, keepdims=True)


def compute_p_val(C, X, Y=None, num_terms=6, p_norm='max', n_tests=100):

    gt = C
    count = 0

    n, dx = X.shape
    for i in tqdm(range(n_tests)):

        # Used to shuffle data
        random_noise = torch.randn(n, dx, device=X.device)
        permutations = torch.argsort(random_noise, dim=0)
        X_permuted = X[permutations, torch.arange(dx).expand(n, dx)]        

        if Y is not None:
            n, dy = Y.shape
            random_noise = torch.randn(n, dy, device=Y.device)
            permutations = torch.argsort(random_noise, dim=0)
            Y_permuted = Y[permutations, torch.arange(dy).expand(n, dy)]        
            null = compute_IDS_torch(X_permuted, Y=Y_permuted, num_terms=num_terms, p_norm=p_norm)
        else:
            null = compute_IDS_torch(X_permuted, Y=Y, num_terms=num_terms, p_norm=p_norm)


        count += torch.where(null > gt, 1, 0)

    p_vals = count / n_tests
    return p_vals


def compute_IDS_torch(X, Y=None, num_terms=6, p_norm='max', p_val=False, num_tests=100):
    n, dx = X.shape
    X_t = transform(X, num_terms=num_terms)
    X_t = center(X_t)

    if Y is not None:
        _, dy = Y.shape
        Y_t = transform(Y, num_terms=num_terms)
        Y_t = center(Y_t)        
        cov = X_t.T @ Y_t
        X_std = torch.sqrt(torch.sum(X_t**2, axis=0))
        Y_std = torch.sqrt(torch.sum(Y_t**2, axis=0))        
        correlations = cov / (X_std.reshape(-1, 1) + EPSILON)
        C = correlations / (Y_std.reshape(1, -1) + EPSILON)
        C = C.reshape(num_terms, dx, num_terms, dy)
    else: 
        C = torch.corrcoef(X_t.T)
        C = C.reshape(num_terms, dx, num_terms, dx)

    C = torch.nan_to_num(C, nan=0, posinf=0, neginf=0)
    C = torch.abs(C)

    if p_norm == 'max':
        C = torch.amax(C, dim=(0, 2))
    elif p_norm == 2:
        C = C**2
        C = torch.mean(C, axis=0)
        C = torch.mean(C, axis=1)
        C = torch.sqrt(C)    
    elif p_norm == 1:
        C = torch.mean(C, axis=0)
        C = torch.mean(C, axis=1)

    if p_val:
        p_vals = compute_p_val(C, X, Y=Y, num_terms=num_terms, p_norm=p_norm, n_tests=num_tests)
        return C, p_vals
    else: 
        return C


if __name__ == "__main__":
    n = 1000
    d1 = 10
    d2 = 2

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X = torch.randn(n, d1, device=device)
    Y = torch.randn(n, d2, device=device)
    Y[:, 0] = torch.sin(X[:, 0])
    Y[:, 1] = torch.cos(X[:, 3])
    # Y = Y.reshape(-1, 1)
    # C = compute_IDS(X, Y, num_terms=6, p_norm='max')
    # print(C)

    C, p_vals = compute_IDS_torch(X, Y, num_terms=6, p_norm='max', p_val=True, num_tests=1000)
    print(C.shape)
    print(C)
    print(p_vals)