"""Python core json  utilitites"""
import json


def check_json(content: dict, val: str, complaint: str) -> json:
    """
    Check if value exists in json or raise a custom error

    Parameters
    ----------

    content : dict
       json to check

    val : str
       value to check for in json

    complaint : str
       Error to raise

    Returns
    -------

    json|dict json that has the content checked for
        nil

    Raises
    ------

    ValueError
       Value error with custom {complaint}

    """
    try:
        result = content[val]
    except Exception as e:
        raise ValueError(complaint)
    return result
