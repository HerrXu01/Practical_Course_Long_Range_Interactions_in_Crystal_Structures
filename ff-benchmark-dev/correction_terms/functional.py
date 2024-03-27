import math
import torch

# Auxiliary Functions

def softplus_inverse(x: torch.Tensor) -> torch.Tensor:
    """
    Inverse of the softplus function. This is useful for initialization of
    parameters that are constrained to be positive (via softplus).
    """
    if not isinstance(x, torch.Tensor):
        x = torch.tensor(x)
    return x + torch.log(-torch.expm1(-x))

def switch_function(x: torch.Tensor, cuton: float, cutoff: float) -> torch.Tensor:
    """
    Switch function that smoothly (and symmetrically) goes from f(x) = 1 to
    f(x) = 0 in the interval from x = cuton to x = cutoff. For x <= cuton,
    f(x) = 1 and for x >= cutoff, f(x) = 0. This switch function has infinitely
    many smooth derivatives.
    NOTE: The implementation with the "_switch_component" function is
    numerically more stable than a simplified version, it is not recommended 
    to change this!
    """
    x = (x - cuton) / (cutoff - cuton)
    ones = torch.ones_like(x)
    zeros = torch.zeros_like(x)
    fp = _switch_component(x, ones, zeros)
    fm = _switch_component(1 - x, ones, zeros)
    return torch.where(x <= 0, ones, torch.where(x >= 1, zeros, fm / (fp + fm)))

def _switch_component(
    x: torch.Tensor, ones: torch.Tensor, zeros: torch.Tensor
) -> torch.Tensor:
    """ Component of the switch function, only for internal use. """
    x_ = torch.where(x <= 0, ones, x)  # prevent nan in backprop
    return torch.where(x <= 0, zeros, torch.exp(-ones / x_))