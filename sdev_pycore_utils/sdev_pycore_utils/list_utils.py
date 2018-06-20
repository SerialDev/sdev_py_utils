"""Python core list datastructure utilitites"""

#from itertools import zip_longest # for Python 3.x
# import zip_longest

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
    return ','.join(str(e) for e in input)



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
        yield l[i:i + n]

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
    return zip_longest(*[iter(iterable)]*n, fillvalue=padvalue)


def uniquify_list_transform(seq, idfun=None): # Alex Martelli ******* order preserving
    if idfun is None:
        def idfun(x): return x
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


def uniquify_list(seq): # Dave Kirby
    # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


def not_in_list(lst, val):
    return [x for x in lst if val not in x ]

# for "pairs" of any length
def chunkwise(t, size=2):
    it = iter(t)
    return zip(*[it]*size)

def chunkwise_window(t):
    for x, y in zip(t, t[1:]):
        yield (x, y)


def fastest_argmax(array):
    array = list( array )
    return array.index(max(array))
