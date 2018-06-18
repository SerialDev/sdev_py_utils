"""Python core list datastructure utilitites"""

#from itertools import zip_longest # for Python 3.x
# import zip_longest


def flatten_list_tuples(list_tuples):
    l = []
    [l.extend(row) for row in list_tuples]
    return l


def CompressList(theList):
    return ','.join(str(e) for e in theList)


def uniquify_list(seq): # Dave Kirby
    # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]

def chunks_padded(n, iterable, padvalue=None):
    "grouper(3, 'abcdefg', 'x') --> ('a','b','c'), ('d','e','f'), ('g','x','x')"
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



def flatten_list_tuples(list_tuples):
    l = []
    [l.extend(row) for row in list_tuples]
    return l



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
