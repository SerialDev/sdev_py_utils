"""Python core functions  utilitites"""


def apply_at(func, pos, iterable):
    """
    * ---------------{Function}---------------
    * Apply a function on any iterable . . .
    * ----------------{Returns}---------------
    * ->result     ::Generator   |Gen expression
    * ----------------{Params}----------------
    * : func     ::Func        |function
    * : pos      ::Int or List |position to apply
    * : iterable ::Iter        |Any python iterable
    """

    if type(pos) is int:
        return (func(x) if i == pos else x for (i, x) in enumerate(iterable))
    elif type(pos) is list:
        return (func(x) if i in pos_lst else x for (i, x) in enumerate(iterable))


def apply_at_tup(func, pos_lst, iterable, apply_to_value=True):
    """
    * ---------------{Function}---------------
    * Apply a function on any iterable . . .
    * ----------------{Returns}---------------
    * ->result     ::Generator   |Gen expression
    * ----------------{Params}----------------
    * : func     ::Func        |function
    * : pos      ::Int or List |position to apply
    * : iterable ::Iter        |Any python iterable
    """
    temp = []
    if apply_to_value == True:

        if type(pos_lst) is int:
            for i, x in enumerate(iterable):
                if pos_lst == i:
                    temp.append((x[0], func(x[1])))
                else:
                    temp.append(x)
        if type(pos_lst) is list:
            for i, x in enumerate(iterable):
                if i in pos_lst:
                    temp.append((x[0], func(x[1])))
                else:
                    temp.append(x)
    # Apply func at key
    elif apply_to_value == False:
        if type(pos_lst) is int:
            for i, x in enumerate(iterable):
                if pos_lst == i:
                    temp.append((funx(x[0]), x[1]))
                else:
                    temp.append(x)
        if type(pos_lst) is list:
            for i, x in enumerate(iterable):
                if i in pos_lst:
                    temp.append(func(x[0]), x[1])
                else:
                    temp.append(x)
    return temp


def batchify(data, batch_size, func):
    """
    Process data in batches using a provided function.

    Parameters
    ----------
    data : iterable
        The data to process.
    batch_size : int
        The size of each batch.
    func : callable
        The function to process each batch.

    Returns
    -------
    list
        List of results from processing each batch.
    """
    results = []
    for batch in chunked(data, batch_size):
        result = func(batch)
        results.extend(result)
    return results
