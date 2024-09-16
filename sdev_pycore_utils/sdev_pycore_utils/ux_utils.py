"""Python core ux  utilitites"""


def pprint_dict(
    d,
    key_color1="\033[92m",
    key_color2="\033[93m",
    value_color1="\033[95m",
    value_color2="\033[96m",
    reset="\033[0m",
):
    """
    Pretty print a dictionary with alternating colors.

    Args:
        d (dict): The dictionary to be printed.
        key_color1 (str, optional): The first key color. Defaults to bright red ('\033[91m').
        key_color2 (str, optional): The second key color. Defaults to bright yellow ('\033[93m').
        value_color1 (str, optional): The first value color. Defaults to bright magenta ('\033[95m').
        value_color2 (str, optional): The second value color. Defaults to bright cyan ('\033[96m').
        reset (str, optional): The reset code. Defaults to '\033[0m'.

    Usage:
        >>> d = {'a': 1, 'b': 2, 'c': 3, 'd': 4, 'e': 5}
        >>> pretty_print_dict(d)
        a : 1
        b : 2
        c : 3
        d : 4
        e : 5
    """
    for i, (key, value) in enumerate(d.items()):
        key_color = key_color1 if i % 2 == 0 else key_color2
        value_color = value_color1 if i % 2 == 0 else value_color2
        print(f"{key_color}{key}{reset} : {value_color}{value}{reset}")


def pprint_model_layers(
    model,
    key_color1="\033[92m",
    key_color2="\033[93m",
    value_color1="\033[95m",
    value_color2="\033[96m",
    reset="\033[0m",
    level=0,
):
    import numpy as np
    import torch
    import mlx.core as mx

    # Handle numpy arrays, PyTorch tensors, and MLX arrays
    if isinstance(model, (np.ndarray, torch.Tensor, mx.array)):
        model = model.cpu().numpy() if isinstance(model, torch.Tensor) else model
        array_str = (
            f"[{', '.join(map(str, model[:2]))} ...]" if model.size > 2 else str(model)
        )
        print(f"{' ' * level}{value_color1}{array_str}{reset}")
    elif isinstance(model, dict):
        for name, module in model.items():
            print(f"{key_color1}{' ' * level}{name}{reset} :")
            pprint_model_layers(
                module,
                key_color2,
                key_color1,
                value_color2,
                value_color1,
                reset,
                level + 2,
            )
    elif isinstance(model, list):
        for idx, item in enumerate(model):
            print(f"{key_color1}{' ' * level}[{idx}]{reset} :")
            pprint_model_layers(
                item,
                key_color2,
                key_color1,
                value_color2,
                value_color1,
                reset,
                level + 2,
            )
    else:
        print(f"{' ' * level}{value_color1}{model}{reset}")
