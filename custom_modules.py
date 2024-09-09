"""
My Custom experimental additions to the paper architecture
- RoPE(Rotary posiitonal embeddings)
- SwiGLU [Done]
- Sparse transformer
"""

from torch.nn import functional as func_nn
from torch import nn


class RotaryEmbeddings(nn.Module):
    pass


class SwiGLU(nn.Module):
    """
    Swish + GLU(Gated Linear Unit activation.
    Proposed by Noam Shazeer in the paper: GLU variants improve transformers.
    """

    def __init__(self, w_param, beta, b_param):
        self.w_param = w_param
        self.beta = beta
        self.b_param = b_param

    def forward(self, x_value):
        learn_w = func_nn.linear(x_value, self.w_param.weight)
        learn_beta = func_nn.linear(x_value, self.beta.weight)
        swish_hidden = func_nn.silu(learn_w) * learn_beta

        gated = func_nn.linear(swish_hidden, self.b_param.weight)

        return gated


class SparseAttention(nn.Module):
    def __init__(self, embed_dim):
        self.embed_dim = embed_dim

    def forward(self, x):
        return x
