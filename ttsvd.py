import math
import torch

def tt_svd(A, eps=0.01):
    d = A.ndim
    B = [None] * d
    C = A.clone()
    delta = (eps / math.sqrt(d-1))*torch.norm(A)
    ranks = [1] + [None] * (d-1)
    for k in range(1, d):
        C = C.reshape(ranks[k-1] * A.shape[k-1], -1)
        U, S, V = torch.linalg.svd(C, full_matrices=False)
        # truncate SVD with reconstruction error < delta 
        for i in range(1, len(S)+1):
            C_temp = U[:, :i] @ torch.diag_embed(S[:i]) @ V[:i, :]
            E = torch.norm(C - C_temp)
            if E <= delta:
                break

        r = i
        ranks[k] = r
        U = U[:, :r].reshape(-1, A.shape[k-1], r)
        S = S[:r]
        V = V[:r, :].reshape(r, -1)
        B[k-1] = U
        C = S.reshape(-1, 1) * V

    B[d-1] = C.reshape(ranks[d-1], A.shape[d-1], -1)
    return B, ranks
