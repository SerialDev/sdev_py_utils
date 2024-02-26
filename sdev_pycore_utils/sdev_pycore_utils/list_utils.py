"""Python core list datastructure utilitites"""

# from itertools import zip_longest # for Python 3.x
# import zip_longest
from collections import Counter


def flatten_list_tuples(list_tuples):
    """
    Flatten a list ot tuples [(,)...] -> [...]

    Parameters
    ----------

    list_tuples : list
       A list of tuples to flatten

    Returns
    -------

    List
        A list with all the values previously inside tuples
    """
    l = []
    [l.extend(row) for row in list_tuples]
    return l


def compress_list(input):
    """
    Compress a list into a , separated string sequence

    Parameters
    ----------

    input : list
       A list with values | Not pyObj

    Returns
    -------

    Str
        A string with the , separated contents
    """
    return ",".join(str(e) for e in input)


# for "pairs" of any length
def chunkwise(t, size=2):
    """
    Yield successive n-sized chunks casted to an iterator from t

    Parameters
    ----------

    l : list
       A list to chunk

    size : int
       Number of elements per chunk

    Returns
    -------

    Generator
        A generator yielding lists of size n
    """

    it = iter(t)
    return zip(*[it] * size)


def chunks(l, n):
    """
    Yield successive n-sized chunks from l

    Parameters
    ----------

    l : list
       A list to chunk

    n : int
       Number of elements per chunk

    Returns
    -------

    Generator
        A generator yielding lists of size n
    """
    for i in range(0, len(l), n):
        yield l[i : i + n]


# def chunks(l, n):
#     n = max(1, n)
#     return (l[i : i + n] for i in range(0, len(l), n))


def chunks_padded(iterable, n, padvalue=None):
    """
    Generate padded chunks from any iterable

    Parameters
    ----------

    iterable : iter
       Any iterable to pad

    n : int
       Number of elements per chunk

    padvalue : str|int|float
       What to pad with to meet minimum chunksize

    Returns
    -------

    Generator
        A generator yielding a padded list of size n


    Doctest
    -------
    >>> list(chunks_padded("abcdefg", 3, "x"))
    [("a","b","c"), ("d","e","f"),("g","x","x")]
    """
    return zip_longest(*[iter(iterable)] * n, fillvalue=padvalue)


def uniquify_list_transform(seq, idfun=None):  # Alex Martelli ******* order preserving
    """
    * ---------------{Function}---------------
    * Remove duplicates from a list while preserving the order and allowing for a custom transform function
    * ----------------{Returns}---------------
    * -> result    ::List       |A new list with duplicates removed
    * ----------------{Params}----------------
    * : seq        ::List       |The input list to remove duplicates from
    * : idfun      ::Callable   |A transform function for comparison (default is None)
    * ----------------{Usage}-----------------
    * >>> uniquify_list_transform([1, 2, 2, 3, 4, 4, 5], lambda x: x * 2)
    * [1, 2, 3, 4, 5]
    """
    if idfun is None:

        def idfun(x):
            return x

    seen = {}
    result = []
    for item in seq:
        marker = idfun(item)
        # in old Python versions:
        # if seen.has_key(marker)
        # but in new ones:
        if marker not in seen:
            seen[marker] = 1
            result.append(item)

    return result


def uniquify_list(seq):  # Dave Kirby
    """
    * ---------------{Function}---------------
    * Remove duplicates from a list while preserving the order
    * ----------------{Returns}---------------
    * -> result    ::List       |A new list with duplicates removed
    * ----------------{Params}----------------
    * : seq        ::List       |The input list to remove duplicates from
    * ----------------{Usage}-----------------
    * >>> uniquify_list([1, 2, 2, 3, 4, 4, 5])
    * [1, 2, 3, 4, 5]
    """
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


def not_in_list(lst, val):
    """
    * ---------------{Function}---------------
    * Remove elements containing a specific value from a list
    * ----------------{Returns}---------------
    * -> result    ::List       |A new list with elements containing the value removed
    * ----------------{Params}----------------
    * : lst        ::List       |The input list to remove elements from
    * : val        ::Any        |The value to search for and remove
    * ----------------{Usage}-----------------
    * >>> not_in_list(['abc', 'def', 'ghi'], 'b')
    * ['def', 'ghi']
    """
    return [x for x in lst if val not in x]


def chunkwise_window(t):
    """
    * ---------------{Function}---------------
    * Create a generator for sliding window pairs from a list
    * ----------------{Returns}---------------
    * -> result    ::Generator  |Generator yielding tuple pairs from the list
    * ----------------{Params}----------------
    * : t          ::List       |The input list for generating sliding window pairs
    * ----------------{Usage}-----------------
    * >>> list(chunkwise_window([1, 2, 3, 4]))
    * [(1, 2), (2, 3), (3, 4)]
    """
    for x, y in zip(t, t[1:]):
        yield (x, y)


def fastest_argmax(array):
    """
    * ---------------{Function}---------------
    * Find the index of the largest element in a list
    * ----------------{Returns}---------------
    * -> result    ::int        |Index of the largest element
    * ----------------{Params}----------------
    * : array      ::List       |The input list to search for the largest element
    * ----------------{Usage}-----------------
    * >>> fastest_argmax([1, 3, 5, 2, 4])
    * 2
    """
    array = list(array)
    return array.index(max(array))


def compare_hashable(s, t):
    """
    Compare two unordered lists that are hashable in O(n)

    Parameters
    ----------

    s : list
       A unordered list to compare

    t : list
       Unordered list to compare to

    Returns
    -------

    bool
        Whether both lists are equal

    """
    return Counter(s) == Counter(t)


def compare_orderable(s, t):
    """
    Compare two unordered lists that are Orderable in O(n log n)

    Parameters
    ----------

    s : list
       A unordered list to compare

    t : list
       Unordered list to compare to

    Returns
    -------

    bool
        Whether both lists are equal

    """
    return sorted(s) == sorted(t)


def compare_equality(s, t):
    """
    Compare two unordered lists that are
    Not Hashable or Orderable in O(n * n)

    Parameters
    ----------

    s : list
       A unordered list to compare

    t : list
       Unordered list to compare to

    Returns
    -------

    bool
        Whether both lists are equal

    """
    t = list(t)  # make a mutable copy
    try:
        for elem in s:
            t.remove(elem)
    except ValueError:
        return False
    return not t


def transform_points_array(data):
    """
    Transform from a points list to a dict with
    arrays of points
    type-def: [{n: w}, {n: w}] -> {n: [w, w]}

    Parameters
    ----------

    data : list
    A list of points

    Returns
    -------

    dict
        A point of arrays
    """
    temp = {}
    for i in data:
        for k, v in i.items():
            if k in temp:
                temp[k].append(v)
            else:
                temp[k] = []
                temp[k].append(v)
    return temp


def dup_detect(source_list):
    """
    Detect Duplicates

    Parameters
    ----------

    source_list list list to check for duplicates :
       nil

    Returns
    -------

    dict
        a dict containing the duplicate and the indexes it occurs in


    Doctest
    -------
    >>> dup_detect([1,2,1])
    {1: [0, 2]}
    """
    from collections import defaultdict

    D = defaultdict(list)
    for i, item in enumerate(source_list):
        D[item].append(i)
    D = {k: v for k, v in D.items() if len(v) > 1}
    return D


def list_argmax(current):
    """
    * ---------------{Function}---------------
    * Find the index and length of the longest list in a list of lists
    * ----------------{Returns}---------------
    * -> max_index ::int        |Index of the longest list
    * -> max_num   ::int        |Length of the longest list
    * ----------------{Params}----------------
    * : current    ::List[List] |The input list of lists
    * ----------------{Usage}-----------------
    * >>> list_argmax([[1, 2], [1, 2, 3], [1]])
    * (1, 3)
    """
    max_num = 0
    max_index = 0
    for index, row in enumerate(current):
        if len(row) > max_num:
            max_num = len(row)
            max_index = index
    return max_index, max_num


def rolling_list(data):
    """
    Create a rolling list of tuples

    Parameters
    ----------

    data : List
       A list to get the data from

    Returns
    -------

    List
        A list of tuples


    Doctest
    -------
    >>> rolling_list([1,2,3,4])
    [(1, 2), (2, 3), (3, 4), (4, None)]
    """
    data.extend([None])
    result = list(zip(data[:], data[1:]))
    data.pop()
    return result


def rolling_doubly_list(data):
    """
    Create a rolling list of tuples

    Parameters
    ----------

    data : List
       A list to get the data from

    Returns
    -------

    List
        A list of tuples


    Doctest
    -------
    >>> rolling_doubly_list([1,2,3,4])
    [(None, 1, 2), (1, 2, 3), (2, 3, 4), (3, 4, None)]
    """
    temp = [None]
    temp.extend(data)
    temp.extend([None])
    result = list(zip(temp[:], temp[1:], temp[2:]))
    return result


def chunks(l, n):
    """
    * ---------------{Function}---------------
    * Yield successive n-sized chunks from a list
    * ----------------{Returns}---------------
    * -> result    ::Generator  |Generator yielding chunks of the list
    * ----------------{Params}----------------
    * : l          ::List       |The input list to be chunked
    * : n          ::int        |The size of the chunks
    * ----------------{Usage}-----------------
    * >>> list(chunks([1, 2, 3, 4, 5], 2))
    * [[1, 2], [3, 4], [5]]
    """
    n = max(1, n)
    return (l[i : i + n] for i in range(0, len(l), n))


def depth_flatten(array, depth=2):
    """
    * ---------------{Function}---------------
    * Flatten a nested list up to a specified depth
    * ----------------{Returns}---------------
    * -> result    ::List       |The flattened list
    * ----------------{Params}----------------
    * : array      ::List       |The input nested list to be flattened
    * : depth      ::int        |The depth to flatten up to (default: 2)
    * ----------------{Usage}-----------------
    * >>> depth_flatten([[1, [2, 3]], [4, [5, 6]]], 1)
    * [1, 2, 3, 4, 5, 6]
    """
    result = array
    for i in range(depth):
        result = functools.reduce(operator.iconcat, result, [])
    return result



def eliminate_common_elements(list1, list2):
    return list(set(list1) - set(list2))


def retain_common_elements(list1, list2):
    return [element for element in list1 if element in list2]



