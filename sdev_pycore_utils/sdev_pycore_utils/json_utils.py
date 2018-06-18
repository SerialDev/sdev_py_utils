"""Python core json  utilitites"""



def check_json(content: dict, val: str, complaint: str) -> json:
    try:
        result = content[val]
    except Exception as e:
        raise ValueError(complaint)
    return result

