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


import re
import sys
import traceback


def extract_full_traceback():
    exc_type, exc_value, tb = sys.exc_info()
    if not exc_type:
        return "❌ **No Active Exception Found**"

    tb_list = traceback.extract_tb(tb)
    entries = []
    root_cause = None
    exception_type = exc_type.__name__
    error_msg = str(exc_value).strip()
    local_vars = {}

    for frame in tb_list:
        short_path = "/".join(frame.filename.replace("\\", "/").split("/")[-3:])
        entry = f"🔹 **File:** `{short_path}`\n   📍 **Line:** `{frame.lineno}`\n   🔄 **Function:** `{frame.name}`"

        # Get local variables from the frame where the error happened
        if frame == tb_list[-1]:  # Last frame (where exception was raised)
            frame_obj = tb.tb_frame
            local_vars = {
                k: repr(v)
                for k, v in frame_obj.f_locals.items()
                if not k.startswith("__")
            }

        entries.append(entry)
        root_cause = f"📌 **Root Cause:** `{short_path}` (Line `{frame.lineno}`)"

    # 🚦 Severity Level Categorization
    severity_levels = {
        "Critical": [
            "MemoryError",
            "SystemExit",
            "KeyboardInterrupt",
            "RecursionError",
        ],
        "Warning": ["SyntaxError", "ImportError", "IndentationError"],
        "Info": [
            "KeyError",
            "TypeError",
            "ValueError",
            "AttributeError",
            "ZeroDivisionError",
            "IndexError",
        ],
    }
    severity = next(
        (
            level
            for level, errors in severity_levels.items()
            if exception_type in errors
        ),
        "Unknown",
    )

    # 💡 Debugging Tips
    debugging_tips = {
        "KeyError": "🔑 **Tip:** Ensure the dictionary key exists before accessing it.",
        "TypeError": "🔢 **Tip:** Check if your variables have the correct type.",
        "ValueError": "🎭 **Tip:** Verify input format and ensure it's within the expected range.",
        "IndexError": "📏 **Tip:** Ensure list/array indices are within valid bounds.",
        "SyntaxError": "🚨 **Syntax Alert:** Check missing colons, parentheses, or incorrect indentation.",
        "MemoryError": "🛑 **Critical:** Consider optimizing data structures or using generators.",
        "ImportError": "📦 **Tip:** Check module installation and `sys.path` settings.",
        "AttributeError": "⚙️ **Tip:** Ensure the object has the attribute/method before accessing it.",
        "ZeroDivisionError": "➗ **Math Tip:** Check for divisions by zero before performing calculations.",
    }
    tip = debugging_tips.get(
        exception_type, "ℹ️ **No specific debugging tip available.**"
    )

    # 🟢 Include Local Variables (if available)
    local_vars_str = (
        "\n".join([f"🔸 `{k}` = `{v}`" for k, v in local_vars.items()])
        if local_vars
        else "❌ No relevant local variables captured."
    )

    return (
        f"""
# 🚨 **TRACEBACK SUMMARY** 🚨
----------------------------------------
❌ **Error:** `{exception_type}: {error_msg}`
----------------------------------------

🔎 **Exception Type:** `{exception_type}`
🚦 **Severity Level:** `{severity}`
----------------------------------------

📜 **CALL STACK (Newest → Oldest Call):**
"""
        + "\n\n".join(entries)
        + f"""

----------------------------------------
{root_cause}
----------------------------------------

💡 **Debugging Tip:** {tip}
----------------------------------------

🟢 **Captured Variables at Failure Point:**
{local_vars_str}
----------------------------------------
"""
    )
