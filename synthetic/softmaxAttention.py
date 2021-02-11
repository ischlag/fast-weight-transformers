import torch.nn.functional as F
import torch.nn as nn
import numpy as np
import torch

from lib import check
from torch import einsum


class SoftmaxAttentionModel(nn.Module):
    def __init__(self, embedding_size, hidden_size, n_keys, n_values):
        super().__init__()
        self.d = d = hidden_size
        self.e = e = embedding_size
        self.n_values = n_values
        self.n_keys = n_keys
        self.vocab_size = n_keys + n_values
        
        self.embeddingV = nn.Embedding(num_embeddings=self.n_values, embedding_dim=self.n_values)
        self.embeddingK = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=e)

        self.W_k = nn.Linear(e + self.n_values, d)
        self.W_v = nn.Linear(e + self.n_values, d)
        self.W_q = nn.Linear(e, d)

        self.reset_parameters()

    def get_name(self):
        return f"d{self.d}_softmax_cat"
    
    def reset_parameters(self):
        # embeddings
        self.embeddingV.weight = torch.nn.Parameter(torch.eye(self.n_values), requires_grad=False)
        nn.init.normal_(self.embeddingK.weight, mean=0., std=1)

        # attention projections
        nn.init.xavier_uniform_(self.W_v.weight)
        nn.init.xavier_uniform_(self.W_k.weight)
        nn.init.xavier_uniform_(self.W_q.weight)
    
    def forward(self, x, q):
        key_indecies = x[:, 0, :]
        value_indecies = x[:, 1, :]
        bs = value_indecies.shape[0]
        sl = value_indecies.shape[1]
        d, e = self.d, self.e
        check(value_indecies, [bs, sl])
        check(key_indecies, [bs, sl])
        check(q, [bs, 1])
        
        # embed words and triggers and concatenate them
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
        
        # compute Q from q
        Q = self.W_q(q)
        check(Q, [bs, 1, d])
        
        # compute attention coefs
        A = einsum("bli,bni->bln", K, Q) / np.sqrt(d)
        check(A, [bs, sl, 1])
        
        # softmax and weighted values
        A = F.softmax(A, dim=1)  # normalise over keys
        y_hat = einsum("bln,bld->bd", A, V)  # sum weighted values
        check(y_hat, [bs, self.n_values])
        
        return y_hat
