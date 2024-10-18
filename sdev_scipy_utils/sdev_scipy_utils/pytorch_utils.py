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


def parameter_summary(model):
    """
    Prints a detailed summary of the parameters in a PyTorch model.

    Parameters
    ----------
    model : nn.Module
        The PyTorch model.

    Returns
    -------
    None
    """
    print(f"{'Layer':<55} {'Parameters':>15}")
    print("-" * 70)
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        param_count = parameter.numel()
        total_params += param_count
        print(f"{name:<55} {param_count:>15}")
    print("-" * 70)
    print(f"{'Total Trainable Params':<55} {total_params:>15}")
