"""Python core tuple datastructure utilitites"""

import pandas as pd


def map_tuple_gen(func, tup):
    return tuple(func(itup) for itup in tup)


def tuple_to_string(tup):
    """
    Map tuples to a string casting any numeric values

    Parameters
    ----------

    tup : tuple
       A tuple to stringify

    Returns
    -------

    str
        A string with all the contents from the tuple
    """
    return "".join(map(str, tup))


def filter_tuples(tups, filter_list, include=True, key=True):
    """
    * type-def ::[Tup] ::[x'∈X] ::Bool ::Bool -> [Tup]'
    * ---------------{Function}---------------
    * Filter  a list of tuples with a list of filters . . .
    * ----------------{Params}----------------
    * : list of tuples
    * : list of filters
    * : whether to include or exclude tuples
    * : whether to use the key or value
    * ----------------{Returns}---------------
    * filtered list of tuples . . .
    """
    if include and key == True:
        result = [(x, y) for x, y in tups if x in filter_list]
    elif include == False and key == True:
        result = [(x, y) for x, y in tups if x not in filter_list]
    elif include == True and key == False:
        result = [(x, y) for x, y in tups if y in filter_list]
    elif include == False and key == False:
        result = [(x, y) for x, y in tups if y not in filter_list]

    return result


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
        raise Exception("KeyNotFound Error")
    elif value is not None and key is None:
        for index, i in enumerate(iterator):
            if key == i[1]:
                return index
            else:
                pass
        raise Exception("ValueNotFound Error")
    else:
        raise Exception("TooManyArguments Error")


def get_key_tuple_lst(tup_lst, position, key=True, value=True):
    """
    * type-def :: lst[tuples] ::  Int ::  Bool ::  Bool
    * ---------------{Function}---------------
    * Get the key/value from a list of tuples at position . . .
    * ----------------{Params}----------------
    * : | list of tuples
    * : | position to retrieve | return key | return value
    * ----------------{Returns}---------------
    * -> Tuple | -> Key | -> Value . . .
    """
    iterator = iter(tup_lst)
    if key and value == True:
        for i in range(position):
            try:
                result = iterator.__next__()
            except StopIteration as e:
                raise Exception(
                    "position greater than list length: {}".format(len(tup_lst))
                )
    elif key is True and value is False:
        for i in range(position):
            try:
                temp = iterator.__next__()
                result = temp[0]
            except StopIteration as e:
                raise Exception(
                    "position greater than list length: {}".format(len(tup_lst))
                )
    elif value is True and key is False:
        for i in range(position):
            try:
                temp = iterator.__next__()
                result = temp[1]
            except StopIteration as e:
                raise Exception(
                    "position greater than list length: {}".format(len(tup_lst))
                )
    return result


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
        for index, i in enumerate(iterator):
            if key == i[0]:
                return i[1]
            else:
                pass
        raise Exception("KeyNotFound Error")
    elif value is not None and key is None:
        for index, i in enumerate(iterator):
            if value == i[1]:
                return i[0]
            else:
                pass
        raise Exception("ValueNotFoundError")
    else:
        raise Exception("TooManyArguments Error")


def flatten_list_tuples(list_tuples):
    """
    * type-def ::(,) -> []
    * ---------------{Function}---------------
    * Flatten a list of tuples [(,), (,), (,)] -> [] . . .
    * ----------------{Params}----------------
    * : list_tuples | a list of tuples
    * ----------------{Returns}---------------
    * A flattened list containing values in the tuples . . .
    """
    l = []
    [l.extend(row) for row in list_tuples]
    return l
