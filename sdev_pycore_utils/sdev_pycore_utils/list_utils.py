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
    # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


def not_in_list(lst, val):
    return [x for x in lst if val not in x]


def chunkwise_window(t):
    for x, y in zip(t, t[1:]):
        yield (x, y)


def fastest_argmax(array):
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
