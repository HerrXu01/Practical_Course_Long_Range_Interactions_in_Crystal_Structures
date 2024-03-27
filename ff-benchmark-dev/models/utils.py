from torch import nn, Tensor
from typing import Tuple, Optional
import numpy as np
import torch
from torch_scatter import gather_csr, scatter, segment_csr
from torch_geometric.utils.num_nodes import maybe_num_nodes


try:
    import sympy as sym
except ImportError:
    sym = None


class RBFExpansion(nn.Module):
    """Expand interatomic distances with radial basis functions."""

    def __init__(
            self,
            vmin: float = 0,
            vmax: float = 8,
            bins: int = 40,
            lengthscale: Optional[float] = None,
            type: str = "gaussian"
    ):
        """Register torch parameters for RBF expansion."""
        super().__init__()
        self.vmin = vmin
        self.vmax = vmax
        self.bins = bins
        self.register_buffer(
            "centers", torch.linspace(vmin, vmax, bins)
        )
        self.type = type

        if lengthscale is None:
            # SchNet-style
            # set lengthscales relative to granularity of RBF expansion
            self.lengthscale = np.diff(self.centers).mean()
            self.gamma = 1 / self.lengthscale

        else:
            self.lengthscale = lengthscale
            self.gamma = 1 / (lengthscale ** 2)

    def forward(self, distance: torch.Tensor) -> torch.Tensor:
        """Apply RBF expansion to interatomic distance tensor."""
        base = self.gamma * (distance.unsqueeze(-1) - self.centers)
        if self.type == 'gaussian':
            return (-base ** 2).exp()
        elif self.type == 'quadratic':
            return base ** 2
        elif self.type == 'linear':
            return base
        elif self.type == 'inverse_quadratic':
            return 1.0 / (1.0 + base ** 2)
        elif self.type == 'multiquadric':
            return (1.0 + base ** 2).sqrt()
        elif self.type == 'inverse_multiquadric':
            return 1.0 / (1.0 + base ** 2).sqrt()
        elif self.type == 'spline':
            return base ** 2 * (base + 1.0).log()
        elif self.type == 'poisson_one':
            return (base - 1.0) * (-base).exp()
        elif self.type == 'poisson_two':
            return (base - 2.0) / 2.0 * base * (-base).exp()
        elif self.type == 'matern32':
            return (1.0 + 3 ** 0.5 * base) * (-3 ** 0.5 * base).exp()
        elif self.type == 'matern52':
            return (1.0 + 5 ** 0.5 * base + 5 / 3 * base ** 2) * (-5 ** 0.5 * base).exp()
        else:
            raise Exception("No Implemented Radial Basis Method")


class RBF(nn.Module):
    def __init__(self, K, edge_types):
        super().__init__()
        self.K = K
        self.means = nn.parameter.Parameter(torch.empty(K))
        self.temps = nn.parameter.Parameter(torch.empty(K))
        self.mul = nn.Embedding(edge_types, 1)
        self.bias = nn.Embedding(edge_types, 1)
        nn.init.uniform_(self.means, 0, 1)
        nn.init.uniform_(self.temps, 0.001, 1.)
        nn.init.constant_(self.bias.weight, 0)
        nn.init.constant_(self.mul.weight, 1)

    def forward(self, x, edge_types):
        mul = self.mul(edge_types)
        bias = self.bias(edge_types)
        x = mul * x + bias
        mean = self.means.float()
        temp = self.temps.float().abs()
        return ((x - mean).square() * (-temp)).exp().type_as(self.means)


@torch.jit.script
def softmax(src: Tensor, index: Optional[Tensor] = None,
            ptr: Optional[Tensor] = None, num_nodes: Optional[int] = None,
            dim: int = 0) -> Tensor:
    r"""Computes a sparsely evaluated softmax.
    Given a value tensor :attr:`src`, this function first groups the values
    along the first dimension based on the indices specified in :attr:`index`,
    and then proceeds to compute the softmax individually for each group.
    Args:
        src (Tensor): The source tensor.
        index (LongTensor, optional): The indices of elements for applying the
            softmax. (default: :obj:`None`)
        ptr (LongTensor, optional): If given, computes the softmax based on
            sorted inputs in CSR representation. (default: :obj:`None`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
        dim (int, optional): The dimension in which to normalize.
            (default: :obj:`0`)
    :rtype: :class:`Tensor`
    """
    if ptr is not None:
        dim = dim + src.dim() if dim < 0 else dim
        size = ([1] * dim) + [-1]
        ptr = ptr.view(size)
        src_max = gather_csr(segment_csr(src, ptr, reduce='max'), ptr)
        out = (src - src_max).exp()
        out_sum = gather_csr(segment_csr(out, ptr, reduce='sum'), ptr)
    elif index is not None:
        N = maybe_num_nodes(index, num_nodes)
        src_max = scatter(src, index, dim, dim_size=N, reduce='max')
        src_max = src_max.index_select(dim, index)
        out = (src - src_max).exp()
        out_sum = scatter(out, index, dim, dim_size=N, reduce='sum')
        out_sum = out_sum.index_select(dim, index)
    else:
        raise NotImplementedError

    return out / (out_sum + 1e-16)