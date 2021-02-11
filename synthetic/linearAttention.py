import torch.nn as nn
import numpy as np
import torch

from torch import einsum, cat, tanh
from torch.nn.functional import relu, elu, layer_norm
from lib import check

# Helper functions for Performer.
# Adapted from Angelos' implementation.
# https://github.com/idiap/fast-transformers/blob/master/fast_transformers/feature_maps/fourier_features.py
def orthogonal_random_matrix_(rows, columns, device):
    """Generate a random matrix whose columns are orthogonal to each
    other (in groups of size `rows`) and their norms is drawn from the
    chi-square distribution with `rows` degrees of freedom
    (namely the norm of a `rows`-dimensional vector distributed as N(0, I)).
    """
    w = torch.zeros([rows, columns], device=device)
    start = 0
    while start < columns:
        end = min(start+rows, columns)
        block = torch.randn(rows, rows, device=device)
        norms = torch.sqrt(torch.einsum("ab,ab->a", block, block))
        Q, _ = torch.qr(block)  # Q is orthonormal
        
        w[:, start:end] = (
            Q[:, :end-start] * norms[None, :end-start]
        )
        start += rows
    return w


# Given x and projection matrix, compute x'.
def prime(x, proj_matrix):
    # x: shape (B, len, dim)
    # proj_matrix (dim, proj_dim)
    _, m = proj_matrix.shape
    
    # Compute offset in logspace
    norm_x_squared = torch.norm(x, dim=-1).pow(2) * 0.5
    sqrt_m = 0.5 * np.log(m)
    sqrt_2 = 0.5 * np.log(2)
    offset = norm_x_squared + sqrt_m + sqrt_2
    offset = offset.unsqueeze(-1)
    check(offset, [x.shape[0], x.shape[1], 1])
    
    u = torch.matmul(x, proj_matrix)
    pos = torch.exp(u - offset)
    neg = torch.exp(-u - offset)

    # last dim is feat.
    out = torch.cat([pos, neg], dim=-1)
    return out


class PrefixSumLinearAttentionModel(nn.Module):
    # phi function
    # linear: Katharopoulos
    # favor: Choromanski
    # dpfp: Schlag-Irie

    # update rule
    # fwm: Schlag-Munkhdalai
    # ours: Schlag-Irie
    
    def __init__(self, 
                 embedding_size, 
                 hidden_size,
                 n_values,
                 n_keys,
                 attention_type='linear',
                 update_rule='sum',
                 eps=1e-6, 
                 arg=0):
        super().__init__()
        assert attention_type in ['linear', 'tanh', 'favor', 'dpfp']
        assert update_rule in ['sum', 'fwm', 'ours']
        self.attention_type = attention_type
        self.update_rule = update_rule
        self.arg = arg
        self.eps = eps
        self.d = d = hidden_size
        self.e = e = embedding_size

        self.vocab_size = n_values + n_keys
        self.n_values = n_values
        self.n_keys = n_keys

        self.embeddingV = nn.Embedding(num_embeddings=self.n_values, embedding_dim=self.n_values)
        self.embeddingK = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=e)

        self.W_k = nn.Linear(e + self.n_values, d)
        self.W_v = nn.Linear(e + self.n_values, d)
        self.W_q = nn.Linear(e, d)

        # additional weights for update rule
        self.W_f = nn.Linear(e + self.n_values, d)  # "forget-key" map
        self.W_b = nn.Linear(e + self.n_values, 1)  # beta map

        self.reset_parameters()

        # print phi size on model init
        phi_size = -1
        if attention_type == "linear":
            phi_size = d
        elif attention_type == "tanh":
            phi_size = d
        elif attention_type == "favor":
            assert self.arg > 0, "favor requires arg to be > 0"
            phi_size = 2 * self.arg
        elif attention_type == "dpfp":
            assert self.arg > 0, "dpfp requires arg to be > 0"
            phi_size = self.arg * d * 2
        else:
            raise Exception(f"No phi_size implementation for attention_type \"{attention_type}\"")
        print(f"phi size: {int(phi_size)}")

    def reset_parameters(self):
        # embeddings
        self.embeddingV.weight = torch.nn.Parameter(torch.eye(self.n_values), requires_grad=False)
        nn.init.normal_(self.embeddingK.weight, mean=0., std=1)

        # attention projections
        nn.init.xavier_uniform_(self.W_v.weight)
        u = 1 / np.sqrt(self.d)
        nn.init.uniform_(self.W_k.weight, a=-u, b=u)
        nn.init.uniform_(self.W_q.weight, a=-u, b=u)

    def get_name(self):
        if self.attention_type == "linear" or self.attention_type == "tanh":
            return f'd{self.d}_{self.attention_type}_{self.update_rule}'
        elif self.attention_type == "favor":
            return f'd{self.d}_favor{self.arg}_{self.update_rule}'
        elif self.attention_type == "dpfp":
            return f'd{self.d}_{self.attention_type}{self.arg}_{self.update_rule}'
        else:
            raise Exception(f"get_name() not implemented for \"{self.attention_type}\"")

    def string_repr(self):
        txt = "Linear Attention"
        txt += f"\n\tattention type: {self.attention_type}"
        txt += f"\n\tupdate rule: {self.update_rule}"
        txt += f"\n\targ: {self.arg}"
        return txt

    def forward(self, x, q):
        key_indecies = x[:, 0, :]
        value_indecies = x[:, 1, :]
        bs, sl = value_indecies.shape
        d, e = self.d, self.e
        check(value_indecies, [bs, sl])
        check(key_indecies, [bs, sl])
        check(q, [bs, 1])

        # embed words and keys and concatenate them
        with torch.no_grad():
            v_emb = self.embeddingV(value_indecies - self.n_values)
            check(v_emb, [bs, sl, self.n_values])

        k_emb = self.embeddingK(key_indecies)
        check(k_emb, [bs, sl, e])

        x = torch.cat([v_emb, k_emb], dim=-1)
        check(x, [bs, sl, e + self.n_values])

        # embed the query
        q = self.embeddingK(q)
        check(q, [bs, 1, e])

        # compute k,v for each pos in parallel
        K = self.W_k(x)
        V = v_emb
        check(K, [bs, sl, d])
        check(V, [bs, sl, self.n_values])

        # compute optional variables
        betas = self.W_b(x)
        F = self.W_f(x)
        check(betas, [bs, sl, 1])
        check(F, [bs, sl, d])

        # compute Q from q
        Q = self.W_q(q)
        check(Q, [bs, 1, d])
        
        # ===== Linear Attentions =====
        if self.attention_type == 'linear':
            K = elu(K, alpha=1.) + 1.
            Q = elu(Q, alpha=1.) + 1.
            check(K, [bs, sl, d])
            check(Q, [bs, 1, d])

        elif self.attention_type == 'tanh':
            K = tanh(K)
            Q = tanh(Q)
            check(K, [bs, sl, d])
            check(Q, [bs, 1, d])

        elif self.attention_type == 'favor':
            m = self.arg
            # Omega: generate random projection matrix
            Omega = orthogonal_random_matrix_(d, m, x.device)
            check(Omega, [d, m])

            K = prime(K, Omega)
            Q = prime(Q, Omega)
            check(K, [bs, sl, m*2])
            check(Q, [bs, 1, m*2])

        elif self.attention_type == 'dpfp':
            check(K, [bs, sl, d])
            check(Q, [bs, 1, d])
            nu = self.arg
            r = lambda x: relu(x)  # relu or exp

            def dpfp(x, nu):
                x = cat([r(x), r(-x)], dim=-1)
                x_rolled = cat([x.roll(shifts=j, dims=-1)
                                for j in range(1, nu+1)], dim=-1)
                x_repeat = cat([x] * nu, dim=-1)
                return x_repeat * x_rolled

            K = dpfp(K, nu)
            Q = dpfp(Q, nu)

            check(K, [bs, sl, d * 2 * nu])
            check(Q, [bs, 1, d * 2 * nu])
        else:
            raise Exception(f"attention not implemented for \"{self.attention_type}\"")

        # ===== Update Rules =====
        p = Q.shape[-1]
        check(V, [bs, sl, self.n_values])
        check(K, [bs, sl, p])
        check(Q, [bs, 1, p])
        check(betas, [bs, sl, 1])

        if self.update_rule == "sum":
            # sum outerproducts of every v and k
            VK = einsum("blv,blk->bvk", V, K)

            # sum keys to normalise
            Z = K.sum(dim=1)

        elif self.update_rule == "fwm":
            # fast weight memory update rule as done by Schlag et. al. (2021)
            betas = torch.sigmoid(betas)
            check(betas, [bs, sl, 1])

            # first update has no old part
            v = V[:, 0, :]
            k = K[:, 0, :]
            beta = betas[:, 0, :]
            W = einsum("bv,bk->bvk", v, k * beta)

            for i in range(1, sl):
                v = V[:, i, :]
                k = K[:, i, :]
                beta = betas[:, i, :]

                old_v = einsum("bvk,bk->bv", W, k)
                W = W - einsum("bv,bk->bvk", old_v, k)
                new_v = beta * v + (1. - beta) * old_v
                W = W + einsum("bv,bk->bvk", new_v, k)

                scale = relu(W.view(bs, -1).norm(dim=-1) - 1) + 1
                W = W / scale.reshape(bs, 1, 1)
            VK = W

        elif self.update_rule == "ours":
            betas = torch.sigmoid(betas)
            check(betas, [bs, sl, 1])

            W = torch.zeros(bs, self.n_values, p).to(K.device)

            for i in range(0, sl):
                v = V[:, i, :]
                k = K[:, i, :]
                beta = betas[:, i, :]
                n = k.sum(dim=-1, keepdim=True)

                # slow implementation
                v_bar = einsum("bdp,bp->bd", W, k / n)
                W = W - einsum("bd,bp->bdp", v_bar, k / n)

                new_v = beta * v + (1. - beta) * v_bar
                W = W + einsum("bd,bp->bdp", new_v, k / n)
            VK = W

        else:
            raise NotImplementedError("Invalid update_rule: ", self.update_rule)
        check(VK, [bs, self.n_values, p])

        # ===== Inference / Query Memory =====
        if self.update_rule == "sum":
            check(Z, [bs, p])
            new_V = einsum("bvp,blp->blv", VK, Q) / (einsum("bp,blp->bl", Z, Q).unsqueeze(-1) + 1e-6)

        elif self.update_rule == "fwm":
            new_V = einsum("bvp,blp->blv", VK, Q)
            new_V = layer_norm(new_V, [self.n_values, ], weight=None, bias=None)

        elif self.update_rule == "ours":
            n = torch.sum(Q, dim=-1, keepdim=True) + 1e-6
            new_V = einsum("bvp,blp->blv", VK, Q / n)

        check(new_V, [bs, 1, self.n_values])
        y_hat = new_V.squeeze(1)
        check(y_hat, [bs, self.n_values])

        return y_hat
