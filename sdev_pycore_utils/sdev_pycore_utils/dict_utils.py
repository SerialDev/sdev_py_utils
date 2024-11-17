"""Python core dict datastructure utilitites"""


def uniquify_to_dict(value):
    """
    Uniquify values in an iterator into a dictionary
    NOTE: USE C impl for performance 2x, status: DONE
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
            if len(super_dict[k]) == 1:
                super_dict[k] = super_dict[k][0]
        return super_dict
    else:
        super_dict = {}
        for k in set(k for d in dicts for k in d):
            super_dict[k] = [d[k] for d in dicts if k in d]
            if len(super_dict[k]) == 1:
                super_dict[k] = super_dict[k][0]
        return super_dict


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
    """
    * ---------------{Function}---------------
    * Recursively searches for a value corresponding to the specified key in a (possibly nested) dictionary
    * ----------------{Returns}---------------
    * -> result    ::Any        |The value corresponding to the specified key, or None if the key is not found
    * ----------------{Params}----------------
    * : obj        ::Dict       |The (possibly nested) dictionary to search
    * : key        ::Any        |The key to search for in the dictionary
    * ----------------{Usage}-----------------
    * >>> data = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    * >>> _finditem(data, 'c')
    * 2
    * >>> _finditem(data, 'x')
    * None
    """
    if key in obj:
        return obj[key]
    for k, v in obj.items():
        if isinstance(v, dict):
            item = _finditem(v, key)
            if item is not None:
                return item


def search(d, key, default=None):
   """
    * ---------------{Function}---------------
    * Searches for a value corresponding to the specified key in a (possibly nested) dictionary
    * ----------------{Returns}---------------
    * -> result    ::Any        |The value corresponding to the specified key, or the default value if the key is not found
    * ----------------{Params}----------------
    * : d          ::Dict       |The (possibly nested) dictionary to search
    * : key        ::Any        |The key to search for in the dictionary
    * : default    ::Any        |The default value to return if the key is not found (default is None)
    * ----------------{Usage}-----------------
    * >>> data = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    * >>> search(data, 'c')
    * 2
    * >>> search(data, 'x', default='Not Found')
    * 'Not Found'
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


def type_executor(tuple_list, data_param):
    """
    * ---------------{Function}---------------
    * Executes a function based on the input data type from a list of type-function pairs
    * ----------------{Returns}---------------
    * -> result    ::Dict       |A dictionary with a single key-value pair, where the key is the type of the data and the value is the result of executing the corresponding function
    * ----------------{Params}----------------
    * : tuple_list ::List[Tuple[Type, Callable]]|A list of tuples where each tuple contains a data type and a corresponding function to execute
    * : data_param ::Any        |The input data to be processed by the appropriate function based on its type
    * ----------------{Usage}-----------------
    * >>> def double(x):
    * ...     return x * 2
    * >>> def to_upper(x):
    * ...     return x.upper()
    * >>> type_executors = [(int, double), (str, to_upper)]
    * >>> type_executor(type_executors, 5)
    * {'int': 10}
    * >>> type_executor(type_executors, 'hello')
    * {'str': 'HELLO'}
    """
    import re

    result = {}
    modified = False
    for current_type, current_fn in tuple_list:
        if type(data_param) == current_type:
            if type(data_param) == type(None):
                continue

            result[
                re.compile("'.*'").findall(str(current_type))[0].replace("'", "")
            ] = current_fn(data_param)

            modified = True
    if modified == False:
        result["unmodified"] = []
        result["unmodified"].append(data_param)
    return result


def print_return(x, key=""):
    """
    * ---------------{Function}---------------
    * Returns a string formed by concatenating a given key and a value separated by an underscore
    * ----------------{Returns}---------------
    * -> result    ::str        |A string containing the key and value separated by an underscore
    * ----------------{Params}----------------
    * : x          ::Any        |The value to be concatenated with the key
    * : key        ::str        |The key to be concatenated with the value (default is an empty string)
    * ----------------{Usage}-----------------
    * >>> print_return('value', 'key')
    * 'key_value'
    """
    if key == None:
        key = ""
    # print(x)
    return key + "_" + x


def identity(x):
    """
    * ---------------{Function}---------------
    * Identity function that returns its input unchanged
    * ----------------{Returns}---------------
    * -> result    ::Any        |The input value
    * ----------------{Params}----------------
    * : x          ::Any        |The value to be returned
    * ----------------{Usage}-----------------
    * >>> identity(42)
    * 42
    """
    return x


def flatten_dict(current_dict, key=None):
    """
    * ---------------{Function}---------------
    * Flattens a nested dictionary into a single dictionary
    * ----------------{Returns}---------------
    * -> result    ::Dict       |A flattened dictionary containing the combined key-value pairs from the input nested dictionary
    * ----------------{Params}----------------
    * : current_dict ::Dict      |A nested dictionary to be flattened
    * : key          ::Any       |An optional key used for handling specific cases (default is None)
    * ----------------{Usage}-----------------
    * >>> data = {'a': 1, 'b': {'c': 2, 'd': {'e': 3}}}
    * >>> flatten_dict(data)
    * {'a': 1, 'c': 2, 'e': 3}
    """
    flattened_dict = {}
    for key in current_dict.keys():
        type_executors = [
            (str, lambda x: print_return(x, key)),
            (dict, lambda x: flatten_dict(x, key=key)),
            (list, lambda x: flatten_dict_list(x, key=key)),
            (None, identity),
        ]
        u = type_executor(type_executors, current_dict[key])
        try:
            flattened_dict = merge_dictionaries([flattened_dict, u["list"]])
        except KeyError as e:
            pass
        try:
            flattened_dict = merge_dictionaries([flattened_dict, u["dict"]])
        except KeyError as e:
            pass
        try:
            flattened_dict[u["str"].replace(" ", "-").split("_")[0]] = (
                u["str"].replace(" ", "-").split("_")[1]
            )
        except KeyError as e:
            pass

    return flattened_dict


def flatten_dict_list(current_list, key=None):
    """
    * ---------------{Function}---------------
    * Flattens a list of dictionaries into a single dictionary
    * ----------------{Returns}---------------
    * -> result    ::Dict       |A flattened dictionary containing the combined key-value pairs from the input list of dictionaries
    * ----------------{Params}----------------
    * : current_list ::List[Dict]|A list of dictionaries to be flattened
    * : key          ::Any       |An optional key used for handling specific cases (default is None)
    * ----------------{Usage}-----------------
    * >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'c': 4}]
    * >>> flatten_dict_list(data)
    * {'a': 3, 'b': 2, 'c': 4}
    """
    flattened_dict = {}
    type_executors = [
        (str, lambda x: print_return(x, key)),
        (dict, lambda x: flatten_dict(x, key=key)),
        (list, lambda x: flatten_dict_list(x, key=key)),
        (None, identity),
    ]

    for i in current_list:
        u = type_executor(type_executors, i)
        try:
            flattened_dict = merge_dictionaries([flattened_dict, u["dict"]])
        except KeyError as e:
            pass
        try:
            flattened_dict[u["str"].split("_")[0]] = (
                u["str"].replace(" ", "-").split("_")[1]
            )
        except KeyError as e:
            pass

    return flattened_dict


def deconstruct(dict_list, key):
    """
    * ---------------{Function}---------------
    * Extracts a specific key's values from a list of dictionaries
    * ----------------{Returns}---------------
    * -> result    ::List       |A list containing the values of the specified key
    * ----------------{Params}----------------
    * : dict_list  ::List[Dict] |A list of dictionaries
    * : key        ::str        |The key to extract the values from
    * ----------------{Usage}-----------------
    * >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4}, {'a': 5, 'b': 6}]
    * >>> deconstruct(data, 'a')
    * [1, 3, 5]
    """
    return list(map(lambda x: x[key], dict_list))


def multi_deconstruct(dict_list, key_list):
    """
    * ---------------{Function}---------------
    * Extracts multiple keys' values from a list of dictionaries
    * ----------------{Returns}---------------
    * -> result    ::List[List] |A list of lists containing the values of the specified keys
    * ----------------{Params}----------------
    * : dict_list  ::List[Dict] |A list of dictionaries
    * : key_list   ::List[str]  |A list of keys to extract the values from
    * ----------------{Usage}-----------------
    * >>> data = [{'a': 1, 'b': 2, 'c': 3}, {'a': 4, 'b': 5, 'c': 6}, {'a': 7, 'b': 8, 'c': 9}]
    * >>> multi_deconstruct(data, ['a', 'c'])
    * [[1, 3], [4, 6], [7, 9]]
    """
    query = "lambda x:["
    for i in key_list:
        query += f"x['{i}'],"
    query = query
    query += "]"
    return list(map(eval(query), dict_list))

def get_by_path(root, items):
    """
    * ---------------{Function}---------------
    * Access a nested object in root by item sequence
    * ----------------{Returns}---------------
    * -> result    ::Any        |The value in the nested object specified by the item sequence
    * ----------------{Params}----------------
    * : root       ::Dict       |The root dictionary containing the nested object
    * : items      ::List       |The sequence of keys to access the nested object
    * ----------------{Usage}-----------------
    * >>> data = {'a': {'b': {'c': 42}}}
    * >>> get_by_path(data, ['a', 'b', 'c'])
    * 42
    """
    from functools import reduce  # forward compatibility for Python 3
    import operator

    return reduce(operator.getitem, items, root)


def set_by_path(root, items, value):
    """
    * ---------------{Function}---------------
    * Set a value in a nested object in root by item sequence
    * ----------------{Params}----------------
    * : root       ::Dict       |The root dictionary containing the nested object
    * : items      ::List       |The sequence of keys to access the nested object
    * : value      ::Any        |The value to set in the nested object
    * ----------------{Usage}-----------------
    * >>> data = {'a': {'b': {'c': 42}}}
    * >>> set_by_path(data, ['a', 'b', 'c'], 13)
    * >>> data
    * {'a': {'b': {'c': 13}}}
    """
    get_by_path(root, items[:-1])[items[-1]] = value


def del_by_path(root, items):
    """
    * ---------------{Function}---------------
    * Delete a key-value in a nested object in root by item sequence
    * ----------------{Params}----------------
    * : root       ::Dict       |The root dictionary containing the nested object
    * : items      ::List       |The sequence of keys to access the nested object
    * ----------------{Usage}-----------------
    * >>> data = {'a': {'b': {'c': 42}}}
    * >>> del_by_path(data, ['a', 'b', 'c'])
    * >>> data
    * {'a': {'b': {}}}
    """
    del get_by_path(root, items[:-1])[items[-1]]


def flatten_dictionary(d, parent_key="", sep="_"):
    """
    NOTE: USE C impl for performance 6x, status: DONE
    * ---------------{Function}---------------
    * Flatten a nested dictionary into a single-level dictionary
    * ----------------{Returns}---------------
    * -> result    ::Dict       |The flattened dictionary
    * ----------------{Params}----------------
    * : d          ::Dict       |The nested dictionary to flatten
    * : parent_key ::str        |The parent key, used internally for recursion (default: "")
    * : sep        ::str        |The separator to use when concatenating keys (default: "_")
    * ----------------{Usage}-----------------
    * >>> data = {'a': {'b': {'c': 42}}}
    * >>> flatten_dictionary(data)
    * {'a_b_c': 42}
    """
    import collections

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

