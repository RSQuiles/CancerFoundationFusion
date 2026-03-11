import torch
from torch.autograd import Function
from typing import cast


class GradientReverse(Function):
    """
    Implements a custom autograd function for gradient reversal.

    This function acts as an identity function during the forward pass but reverses
    and scales the gradient during the backward pass. It's a core component for
    implementing Domain-Adversarial Neural Networks (DANNs), where it's used to
    train a feature extractor to produce domain-invariant features.
    """

    scale = 1.0

    @staticmethod
    def forward(ctx, x: torch.Tensor) -> torch.Tensor:
        """Performs the forward pass (identity operation)."""
        return x

    @staticmethod
    def backward(ctx, *grad_outputs: torch.Tensor) -> torch.Tensor:
        """Performs the backward pass (gradient reversal and scaling).

        Args:
            ctx: A context object with saved information from the forward pass.
            *grad_outputs (torch.Tensor): The gradient from the subsequent layer.

        Returns:
            Tuple[None, torch.Tensor]: A tuple containing None (for the `ctx` argument)
            and the reversed, scaled gradient.
        """
        grad_output = grad_outputs[0]
        return GradientReverse.scale * grad_output.neg()


def grad_reverse(x: torch.Tensor, scale: float) -> torch.Tensor:
    """
    Applies the gradient reversal operation to a tensor.

    This is a functional interface for the `GradientReverse` layer.

    Args:
        x (torch.Tensor): The input tensor.
        scale (float, optional): The factor by which to scale the reversed
            gradient. Defaults to 1.0.

    Returns:
        torch.Tensor: The output tensor, which is identical to the input but will
        have its gradient reversed during backpropagation.
    """
    GradientReverse.scale = scale
    return cast(torch.Tensor, GradientReverse.apply(x))
