"""Python core dict datastructure utilitites"""

def uniquify_to_dict(value):
    result = {}
    temp = []
    current = ''
    for x, y in value:
        if x == current:
            temp.append(y)
        else:
            result[current] = temp
            temp = []
            current = x
            temp.append(y)
        result[current] = temp

    return {k: v for k, v in result.items() if k is not ''}



