"""Python core tuple datastructure utilitites"""

import pandas as pd

def filter_tuples(tups, filter_list, include=True):
    if include == True:
        result = [(x, y) for x, y in tups if x in filter_list]
    else:
        result = [(x, y) for x, y in tups if x not in filter_list]
    return result


def tuples_to_pd(tup):
    temp = pd.DataFrame(tup).T

    return pd_row_header(temp)


def get_position_tuple_lst(tup_lst, key=None, value=None):
    """
    * type-def :: [Tuples] ->  Key ->  Value
    * ---------------{Function}---------------
    * Efficiently iterate through large lists of tuples . . .
    * ----------------{Params}----------------
    * : list of tuples
    * : key to match
    * : value to match
    * ----------------{Returns}---------------
    * Position where key was found . . .
    """
    iterator = iter(tup_lst)
    if key is not None and value is None:
        for index, i in enumerate(iterator):
            if key == i[0]:
                return index
            else:
                pass
        raise Exception('KeyNotFound Error')
    elif value is not None and key is None:
        for index, i in enumerate(iterator):
            if value == i[1]:
                return index
            else:
                pass
        raise Exception('ValueNotFound Error')
    else:
        raise Exception('TooManyArguments Error')

def get_matching_tuple_lst(tup_lst, key=None, value=None):
    """
    * type-def :: [Tuple] ::  x'∈X ::  x'∈X ->  x'∈[Tuple]
    * ---------------{Function}---------------
    * Get the matching key/value from a list of tuples . . .
    * ----------------{Params}----------------
    * : | list of tuples
    * : | key to query | value to query
    * ----------------{Returns}---------------
    * the first key matching a value or first value matching a key . . .
    """
    iterator = iter(tup_lst)
    if key is not None and value is None:
        for index, i in enumerate (iterator):
            if key == i[0]:
                return i[1]
            else:
                pass
        raise Exception('KeyNotFound Error')
    elif value is not None and key is None:
        for index, i in enumerate(iterator):
            if value == i[1]:
                return i[0]
            else:
                pass
        raise Exception('ValueNotFoundError')
    else:
        raise Exception('TooManyArguments Error')
