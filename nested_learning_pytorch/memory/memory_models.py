import torch
from torch import nn, cat
import torch.nn.functional as F
from torch.nn import Module, Parameter, ParameterList

from einops import einsum

from typing import Callable

# functions

def l2norm(t):
    return F.normalize(t, dim = -1)


class Normalization(Module):
    def __init__(self, **kwargs):
        super().__init__()
        
    def forward(self, x):
        raise 

class NormalizationBuilder(Callable):
    def __init__(self, normalization_type: str):
        self.normalization_type = normalization_type
        
    def __call__(self, *args, **kwargs) -> Normalization:
        if self.normalization_type == 'pre_norm_only':
            return PreNormOnly(*args, **kwargs)
        elif self.normalization_type == 'post_norm_only':
            return PostNormOnly(*args, **kwargs)
        elif self.normalization_type == 'residual_pre_norm':
            return ResidualPreNorm(*args, **kwargs)
        elif self.normalization_type == 'residual_post_norm':
            return ResidualPostNorm(*args, **kwargs)
        elif self.normalization_type == 'residual_norm':
            return ResidualNorm(*args, **kwargs)
        elif self.normalization_type is None and 'model' in kwargs:
            return kwargs['model']
        raise ValueError(f'Invalid normalization type: {self.normalization_type}')

# norms

class LayerNorm(Module):
    def __init__(
        self,
        dim,
        is_multi_head = False
    ):
        super().__init__()

        self.ln = nn.LayerNorm(dim, elementwise_affine = False)
        self.gamma = Parameter(torch.zeros(dim))
        
        self.target_ndim = 3 if is_multi_head else 2

    def forward(self, x):
        gamma = self.gamma

        if gamma.ndim != self.target_ndim:
            expand_dims = self.target_ndim - gamma.ndim
            for _ in range(expand_dims):
                gamma = gamma.unsqueeze(0)

        return self.ln(x) * (gamma + 1.)

class PreNormOnly(Normalization):
    def __init__(
        self,
        dim,
        model: Module,
        is_multi_head = False,
        **kwargs
    ):
        super().__init__()
        
        self.ln = LayerNorm(dim, is_multi_head=is_multi_head)
        self.model = model

    def forward(self, x, pattern: str | list[str] | None = None):
        return self.model(self.ln(x), pattern=pattern)

class PostNormOnly(Normalization):
    def __init__(
        self,
        dim,
        model: Module,
        is_multi_head = False,
        out_dim = None,
        **kwargs
    ):
        super().__init__()
        
        self.ln = LayerNorm(dim if out_dim is None else out_dim, is_multi_head=is_multi_head)
        self.model = model

    def forward(self, x, pattern: str | list[str] | None = None):
        return self.ln(self.model(x, pattern=pattern))

class ResidualPreNorm(Normalization):
    def __init__(
        self,
        dim,
        model: Module,
        is_multi_head = False,
        out_dim = None,
        **kwargs
    ):
        super().__init__()
        assert out_dim is None or out_dim == dim, "out_dim is not supported for ResidualPreNorm: x can't add with an output with a different dimension"
        
        self.ln = LayerNorm(dim, is_multi_head=False)
        self.model = model
        
        self.target_ndim = 3 if is_multi_head else 2

    def forward(self, x, pattern: str | list[str] | None = None):

        out = self.model(self.ln(x), pattern=pattern)
        
        if x.ndim != self.target_ndim:
            expand_dims = self.target_ndim - x.ndim
            for _ in range(expand_dims):
                x = x.unsqueeze(0)
        return x + out

class ResidualPostNorm(Normalization):
    def __init__(
        self,
        dim,
        model: Module,
        is_multi_head = False,
        out_dim = None,
        **kwargs
    ):
        super().__init__()
        assert out_dim is None or out_dim == dim, "out_dim is not supported for ResidualPostNorm: x can't add with an output with a different dimension"
        
        self.ln = LayerNorm(dim if out_dim is None else out_dim, is_multi_head=is_multi_head)
        self.model = model
        
        self.target_ndim = 3 if is_multi_head else 2

    def forward(self, x, pattern: str | list[str] | None = None):

        out = self.model(x, pattern=pattern)
        
        if x.ndim != self.target_ndim:
            expand_dims = self.target_ndim - x.ndim
            for _ in range(expand_dims):
                x = x.unsqueeze(0)
        return self.ln(x + out)


# norm + residual wrapper, as used in original TTT paper
class ResidualNorm(Normalization):
    def __init__(
        self,
        dim,
        model: Module,
        is_multi_head = False,
        out_dim = None,
        **kwargs
    ):
        super().__init__()
        assert out_dim is None or out_dim == dim, "out_dim is not supported for ResidualNorm: x can't add with an output with a different dimension"
        
        self.ln = LayerNorm(dim, is_multi_head=is_multi_head)
        self.model = model
        
        self.target_ndim = 3 if is_multi_head else 2

    def forward(self, x, pattern: str | list[str] | None = None):

        out = self.model(x, pattern=pattern)
        
        if x.ndim != self.target_ndim:
            expand_dims = self.target_ndim - x.ndim
            for _ in range(expand_dims):
                x = x.unsqueeze(0)
        return self.norm(out) + x

# memory mlp proposed in TTT

class MemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        is_multi_head = False,
        n_heads = 1,
        expansion_factor = 2.,
        out_dim = None,
        with_bias = False,
        activation_fn = nn.GELU()
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)
        dim_out = dim if out_dim is None else out_dim
        dims = (dim, *((dim_hidden,) * (depth - 1)), dim_out)

        if is_multi_head:
            self.weights = ParameterList([
                Parameter(torch.randn(n_heads, dim_in, dim_out)) 
                for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ])
            if with_bias:
                self.biases = ParameterList([
                    Parameter(torch.zeros(n_heads, dim_out)) for dim_out in dims[1:]
                ])

            # Use Kaiming uniform initialization, considering head dimension
            for weight in self.weights:
                # Initialize each head independently
                fan_in = weight.shape[1]  # dim_in
                fan_out = weight.shape[2]  # dim_out
                gain = nn.init.calculate_gain('relu')
                std = gain / torch.sqrt(torch.tensor(fan_in, dtype=torch.float32))
                bound = torch.sqrt(torch.tensor(3.0)) * std
                nn.init.uniform_(weight, -bound, bound)
        else:
            self.weights = ParameterList([
                Parameter(torch.randn(dim_in, dim_out)) for dim_in, dim_out in zip(dims[:-1], dims[1:])
            ])
            if with_bias:
                self.biases = ParameterList([
                    Parameter(torch.zeros(dim_out)) for dim_out in dims[1:]
                ])

            for weight in self.weights:
                nn.init.xavier_uniform_(weight)
        
        self.with_bias = with_bias
        self.activation_fn = activation_fn

    def forward(
        self,
        x,
        pattern: str | list[str] | None = None
    ):
        if isinstance(pattern, str):
            pattern = [pattern] * len(self.weights)
        elif isinstance(pattern, list):
            assert len(pattern) == len(self.weights), 'pattern must be a list of the same length as the number of weights'
        else:
            pattern = [None] * len(self.weights)

        for ind, (weight, p) in enumerate(zip(self.weights, pattern)):
            is_first = ind == 0

            if not is_first:
                x = self.activation_fn(x)

            if p is None:
                x = x @ weight
            else:
                x = einsum(x, weight, p)

            if self.with_bias:
                bias = self.biases[ind]
                while bias.ndim != x.ndim:
                    bias = bias.unsqueeze(0)
                x = x + bias

        return x

# memory mlp, but with gated residual + final projection

class GatedResidualMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        expansion_factor = 4.
    ):
        super().__init__()
        dim_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, dim_hidden)),
                Parameter(torch.randn(dim_hidden, dim)),
                Parameter(torch.randn(dim * 2, dim)),
            ]) for _ in range(depth)
        ])

        self.final_proj = Parameter(torch.randn(dim, dim))

        for param in self.parameters():
            nn.init.xavier_uniform_(param)

    def forward(
        self,
        x
    ):

        for weight1, weight2, to_gates in self.weights:
            res = x

            hidden = x @ weight1
            hidden = F.gelu(hidden)
            branch_out = hidden @ weight2

            # gated residual

            gates = cat((branch_out, res), dim = -1) @ to_gates
            x = res.lerp(branch_out, gates.sigmoid())

        return x @ self.final_proj

# memory mlp with factorized weights
# so can tradeoff capacity for smaller chunk sizes

class FactorizedMemoryMLP(Module):
    def __init__(
        self,
        dim,
        depth,
        k = 32
    ):
        super().__init__()
        self.weights = ParameterList([
            ParameterList([
                Parameter(torch.randn(dim, k)),
                Parameter(torch.randn(k, dim)),
            ]) for _ in range(depth)
        ])

        for weight1, weight2 in self.weights:
            nn.init.xavier_uniform_(weight1)
            nn.init.xavier_uniform_(weight2)

    def forward(
        self,
        x
    ):

        for ind, (weight1, weight2) in enumerate(self.weights):
            is_first = ind == 0

            if not is_first:
                x = F.gelu(x)

            x = x @ weight1 @ weight2

        return x

# an MLP modelled after the popular swiglu ff in modern transformers

class MemorySwiGluMLP(Module):
    def __init__(
        self,
        dim,
        depth = 1, # default to 2 layer MLP from TTT, depth of 2 would be 4 layer MLP, but done as 2 feedforwards with residual
        expansion_factor = 4.
    ):
        super().__init__()

        dim_inner = int(dim * expansion_factor * 2 / 3)

        weights = []

        for _ in range(depth):
            weights.append(ParameterList([
                Parameter(torch.randn(dim, dim_inner * 2)),
                Parameter(torch.randn(dim_inner, dim)),
            ]))

        self.weights = ParameterList(weights)
        self.norm = LayerNorm(dim)

    def forward(self, x):

        for w1, w2 in self.weights:
            residual = x

            x, gates = (x @ w1).chunk(2, dim = -1)

            x = x * F.gelu(gates)

            x = x @ w2

            x = x + residual

        return self.norm(x)

# improvised attention as memory module

class MemoryAttention(Module):
    def __init__(
        self,
        dim,
        scale = 8.,
        expansion_factor = 2.
    ):
        super().__init__()
        self.scale = scale
        dim_ff_hidden = int(dim * expansion_factor)

        self.weights = ParameterList([
            Parameter(torch.randn(dim, dim)), # queries
            Parameter(torch.randn(dim, dim)), # keys
            Parameter(torch.randn(dim, dim)), # values
            Parameter(torch.randn(dim, dim_ff_hidden)), # ff w1
            Parameter(torch.randn(dim_ff_hidden, dim)), # ff w2
        ])

        for weight in self.weights:
            nn.init.xavier_uniform_(weight)

    def forward(self, x):

        wq, wk, wv, ffw1, ffw2 = self.weights

        q = l2norm(x @ wq)
        k = l2norm(x @ wk)
        v = x @ wv

        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            scale = self.scale,
            is_causal = True
        )

        # parallel attention + feedforward block
        # as in PaLM + Gpt-J

        h = F.gelu(x @ ffw1)
        ff_out = h @ ffw2

        return attn_out + ff_out

class MomentumModule(nn.Module):
    """A simple wrapper to make a Parameter compatible with ModuleDict."""
    def __init__(self, momentum: torch.Tensor):
        super().__init__()
        self.momentum = nn.Parameter(momentum)
    
    def forward(self, x, pattern=None):
        return x
