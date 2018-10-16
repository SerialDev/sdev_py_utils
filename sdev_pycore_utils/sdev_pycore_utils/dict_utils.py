"""Python core dict datastructure utilitites"""


def uniquify_to_dict(value):
    """
    Uniquify values in an iterator into a dictionary

    Parameters
    ----------

    value : iterator
       iterator to uniquify

    Returns
    -------

    Dictionary
        A python dictionary containing the uniquified values from iterator

    """
    result = {}
    temp = []
    current = ""
    for x, y in value:
        if x == current:
            temp.append(y)
        else:
            result[current] = temp
            temp = []
            current = x
            temp.append(y)
        result[current] = temp

    return {k: v for k, v in result.items() if k is not ""}


def filter_dict(d, filter_string, remove=True):
    """
    * type-def ::{} :: [] :: Bool -> '{}
    * ---------------{Function}---------------
    * filter dictionary based on a filter_string list . . .
    * ----------------{Params}----------------
    * : dict:: python dictionary
    * : filter_string::list of strings to filter by
    * : remove::Boolean to check keep or remove
    * ----------------{Returns}---------------
    * Filtered Dictionary . . .
    """
    if remove:
        for i in filter_string:
            d = {k: v for (k, v) in d.items() if i not in k}
        return d
    else:
        for i in filter_string:
            d = {k: v for (k, v) in d.items() if i in k}
        return d


def map_dictionary(function, dictionary, keys=True, vals=True):
    """
    * type-def ::f(x) :: {} :: Bool :: Bool -> {}
    * ---------------{Function}---------------
    * Map a function across a dictionary [keys||values] . . .
    * ----------------{Params}----------------
    * : function::py_object_function the function to apply
    * : dictionary::{} dictionary to apply function to
    * : keys::Bool whether to apply along keys
    * : vals::Bool whether to apply along values
    * ----------------{Returns}---------------
    * Dictionary with the function applied to each item in it . . .
    """
    if keys and vals:
        key_map = map(function, dictionary.keys())
        val_map = map(function, dictionary.values())
        return dict(zip(key_map, val_map))
    if keys and not vals:
        key_map = map(function, dictionary.keys())
        return dict(zip(key_map, dictionary.values()))
    if not keys and vals:
        val_map = map(function, dictionary.values())
        return dict(zip(dictionary.keys(), val_map))


def merge_dictionaries(dicts, unique_key=False):
    """
    * type-def ::[dict] -> dict
    * ---------------{Function}---------------
    * merge a list of dictionaries into a one dict [unique optional] . . .
    * ----------------{Params}----------------
    * : list of dicts
    * ----------------{Returns}---------------
    * dictionary [unique optional] . . .
    """
    if unique_key:
        super_dict = {}
        for k in set(k for d in dicts for k in d):
            super_dict[k] = set(d[k] for d in dicts if k in d)
        return super_dict
    else:
        super_dict = {}
        for k in set(k for d in dicts for k in d):
            super_dict[k] = [d[k] for d in dicts if k in d]
        return super_dict


def get_key_position(dictionary, position):
    """
    * Function: Efficiently iterate through large dictionaries
    * (40k x speedup over 10 M item dictionary)
    * -----------{returns}------------
    * key at position . . .
    * ------------{usage}-------------
    * get_key_position(dictionary, 1))
    """
    iterator = iter(dictionary)
    for i in range(position):
        try:
            key = iterator.__next__()
        except StopIteration as e:
            raise Exception(
                "position greater than dictionary length: {} ".format(len(dictionary))
            )
    return key


def get_position_key(dictionary, key):
    """
    * Function: Efficiently iterate through large dictionaries
    * (40k x speedup over 10 M item dictionary)
    * -----------{returns}------------
    * position where key is found . . .
    * ------------{usage}-------------
    * get_key_position(dictionary, 'key'))
    """
    iterator = iter(dictionary)
    for index, i in enumerate(iterator):
        if key == i:
            return index

        else:
            pass
    raise Exception("KeyNotFound Error")


def _finditem(obj, key):
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            item = _finditem(v, key)
            if item is not None:
                return item


def search(d, key, default=None):
    """Return a value corresponding to the specified key in the (possibly
    nested) dictionary d. If there is no item with that key, return
    default.
    """
    stack = [iter(d.items())]
    while stack:
        for k, v in stack[-1]:
            if isinstance(v, dict):
                stack.append(iter(v.items()))
                break
            elif k == key:
                return v
        else:
            stack.pop()
    return default


def without_keys(d, keys):
    """
    Filter keys out of a dictionary

    Parameters
    ----------

    d : Dict
       dict with keys to filter

    keys : list
       list of keys to use as a filter

    Returns
    -------

    Dictionary
        Filtered dictionary
    """
    return {x: d[x] for x in d if x not in keys}


def union_dicts(*args):
    """
    Join multiple dicts together

    Parameters
    ----------

    args : dict
       Dictionaries to join

    Returns
    -------

    Dict
        A dictionary that is the union of *args dicts
    """
    from itertools import chain

    return dict(chain.from_iterable(d.items() for d in args))


def transform_dict_ltuples(data):
    """
    dict -> list(tuples)

    Parameters
    ----------

    data : dict
       A dictionary

    Returns
    -------

    list
        A tuple list
    """

    result = [(a, b) for a, b in data.items()]
    return result
