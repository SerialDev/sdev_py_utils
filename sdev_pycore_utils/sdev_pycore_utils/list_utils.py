"""Python core list datastructure utilitites"""

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

