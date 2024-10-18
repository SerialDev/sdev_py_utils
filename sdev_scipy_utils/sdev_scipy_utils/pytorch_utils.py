import torch.nn as nn


def count_parameters(model):
    """
    Count the number of trainable parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.

    Returns
    -------
    int
        Total number of trainable parameters.
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
