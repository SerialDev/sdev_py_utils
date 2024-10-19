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


def chunked(iterable, chunk_size):
    """
    Split an iterable into chunks of a specified size.

    Parameters
    ----------
    iterable : iterable
        The iterable to split.
    chunk_size : int
        The size of each chunk.

    Returns
    -------
    generator
        A generator yielding chunks.
    """
    it = iter(iterable)
    while True:
        chunk = []
        try:
            for _ in range(chunk_size):
                chunk.append(next(it))
            yield chunk
        except StopIteration:
            if chunk:
                yield chunk
            break


# Context manager to handle logging, processing, and error handling
@contextmanager
def process_chunk_context(chunk_indices, df):
    try:
        temp_df = df.loc[chunk_indices]
        print(
            f"\033[36mStarting processing for chunk with indices: {chunk_indices}\033[0m"
        )  # Section heading
        yield temp_df
        print(
            f"\033[32mSuccessfully processed chunk with indices: {chunk_indices}\033[0m"
        )  # Success completion
    except Exception as e:
        print(
            f"\033[31mError processing chunk with indices: {chunk_indices} - {e}\033[0m"
        )  # Error log
        raise
    finally:
        print(
            f"\033[35mFinished processing chunk with indices: {chunk_indices}\033[0m"
        )  # Informative note


# Function that processes chunks using the context manager and yields each chunk
def process_in_chunks(df, chunk_size):
    # Example of using the generator in a loop
    # for df in process_in_chunks(df_full, 10):  # Adjust chunk size as needed

    iterable = df.index
    for chunk_indices in chunked(iterable, chunk_size):
        with process_chunk_context(chunk_indices, df) as temp_df:
            yield temp_df


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
