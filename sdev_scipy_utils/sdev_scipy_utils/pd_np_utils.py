"""Pandas & np  utility functions"""

import base64
import datetime
import inspect
import json
import math
import threading
from io import StringIO
from math import ceil, floor
from multiprocessing.dummy import Pool as ThreadPool

import numpy as np
import pandas as pd


def pd_to_base64(df):
    """
    base64 encode a pandas dataframe

    Parameters
    ----------

    df : pd.DataFrame
       A pandas dataframe to encode

    Returns
    -------

    bytes
        Base64 encoded pandas dataframe
    """
    return base64.b64encode(df.to_json().encode("utf-8")).decode("ascii")


def pd_series_from_base64(encoded):
    """
    Read a base64 encoded message to yield a pandas dataframe

    Parameters
    ----------

    encoded : bytes
       base 64 encoded message

    Returns
    -------

    pd.DataFrame
        A pandas dataframe with the content present in the base64 encoded dataset
    """
    return pd.DataFrame(
        list(json.loads(base64.b64decode(encoded.encode("ascii"))).values()),
        columns="Values",
    )


def pd_load_tuple(tuple_list):
    df = pd.DataFrame(tuple_list).T
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))
    return df


def pd_concat_list_dict(list_dict):
    for i in range(len(list_dict)):
        if i == 0:
            df = pd_load_tuple(list(list_dict[i].items()))
        else:
            df = df.append(pd_load_tuple(list(list_dict[i].items())))
    return df


def pd_csv_to_io(df):
    import io

    buffer = io.BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    return buffer


def chunk_select_mysql(con, table_name, order_by, stride_length=1000):
    query = f"""
CREATE TEMPORARY TABLE MYCHUNKED{table_name} AS (
  SELECT *
  FROM {table_name}
  ORDER BY {order_by}
);
"""
    # temp_table = pd.read_sql(query, con)
    # row_count = pd.read_sql(f"select Count(*) as row_count from MYCHUNKED{table_name}",mariadb_connection).row_count[0]
    # deleted_temp = pd.read_sql(f"DROP TEMPORARY TABLE IF EXISTS MYCHUNKED{table_name};", con)
    row_count = pd.read_sql(
        f"select Count(*) as row_count from {table_name}", mariadb_connection
    ).row_count[0]
    print(row_count)
    for i in range(0, row_count, stride_length):
        current = i + stride_length
        yield pd.read_sql(f"select * from {table_name} LIMIT {i}, {current};", con)


def get_stride_len(size_data, chunks):
    from math import ceil

    stride = ceil(size_data / chunks)
    return stride


# {Get attributes pandas}#


def get_attributes(mod):
    attributes = inspect.getmembers(mod, lambda a: not (inspect.isroutine(a)))
    return [
        a
        for a in attributes
        if not (a[0].startswith("__") and a[0].endswith("__") or a[0].startswith("_"))
    ]


def get_attributes_pd(obj, filter_list=None):
    if filter_list:
        filtered_attr = filter_tuples(get_attributes(obj), filter_list)
        df = pd.DataFrame(filtered_attr).T
    else:
        df = pd.DataFrame(get_attributes(obj)).T
    return pd_row_header(df)


def get_nested_attributes_pd(data, filter_list=None, axis=0):
    for i, section in enumerate(data):
        if i == 0:
            sections = get_attributes_pd(section, filter_list)
        else:
            temp = get_attributes_pd(section, filter_list)
            sections = pd.concat((sections, temp), axis=axis)
    return sections


def concat_pd_list(data, axis=0):
    for i, section in enumerate(data):
        if i == 0:
            sections = section
        else:
            sections = pd.concat((sections, section), axis=axis)
    return sections


# {Row to column header}#


def pd_row_header(df, idx=0):
    df.columns = df.iloc[idx]
    return df.reindex(df.index.drop(idx))


def pd_split_lazy(df, column):
    return zip(*df[column].str.split().tolist())


# ----------{Tuples to pandas}----------#


def tuples_to_pd(tup):
    """
    Convert a list of tuples to a pandas dataframe

    Parameters
    ----------

    tup : list
       List of tuples

    Returns
    -------

    pd.DataFrame
        A pandas dataframe with tuples[0] as header
    """
    temp = pd.DataFrame(tup).T

    return pd_row_header(temp)


def encode_flags(data, flags, col_to_check):
    """
    * type-def ::pd.DataFrame :: tuple :: str -> pd.DataFrame
    * ---------------{Function}---------------
    * One hot encode column based on flags . . .
    * ----------------{Params}----------------
    * : data | pandas dataframe holding the data : flags | a tuple holding
    * 'col_name': val : col_to_check | column name to one_hot_encode
    * ----------------{Returns}---------------
    * pd.DataFrame holding the encoded data  . . .
    * -----------------{Extra}----------------
    * [MUTABLE STATE WARNING] This will modify a dataframe in place . . .
    """
    for i, val in flags:
        data[i] = 0
        data[i][data[col_to_check] == val] = 1
    return data


# -----{Naive numpy parallelization}----#

# TODO: cap threads and processes depending on platform/length of initial data
def np_parallel(func, data, parts=4, verbose=False):
    """
    Split and apply a function to numpy array

    Parameters
    ----------

    func : function
       A function to apply to the np.array

    data : np.ndarray
       A numpy array to apply the function to

    parts : int
       How many parts to split the operation to

    verbose : bool
       Enable verbose output

    Returns
    -------

    np.array
       Re-constructed numpy array with all the parts having function applied
    """

    def split_length(data, parts=4):
        return np.int(np.ceil(data.shape[0] / parts))

    def split_array(data, parts=4):
        split_len = split_length(data, parts)
        split_array = []
        for index, i in enumerate(range(parts), 1):
            if index == 1:
                array = data[0:split_len]
                split_array.append((array))
            else:
                array = data[(split_len * (i - 1)) : (split_len * i)]
                split_array.append((array))
        return np.array(split_array)

    def np_multi(func, data, parts=4):
        split = split_array(data, parts)
        pool = Pool(parts)
        applied = np.array(pool.map(func, split))
        applied = np.concatenate(([*applied]), axis=0)
        return applied

    if Pool.__module__ == "multiprocessing.dummy":
        if verbose:
            print(
                "executing {func} using multiple({parts}) threads".format(
                    func=func, parts=parts
                )
            )
            return np_multi(func, data, parts)
        else:
            return np_multi(func, data, parts)
    elif Pool.__module__ == "multiprocessing":
        if verbose:
            print(
                "executing {func} using multiple({parts}) processes".format(
                    func=func, parts=parts
                )
            )
            return np_multi(func, data, parts)
        else:
            return np_multi(func, data, parts)


# --{Slice between}--#


def get_between(data, col=None, _from=None, _to=None):
    """
    Get data between two values

    Parameters
    ----------

    data : pd.DataFrame | np.array
        Data to slice using boolean indexing

    col : str(Optional)
       Column to make the slice based on (Pandas only)

    _from : int|float
       What value to slice from

    _to : int|float
       What value to slice to

    Returns
    -------

    pd.DataFrame|np.array
         Returns a slice of the data in whichever format it came from
    """

    def np_get_between(ndarray, _from=None, _to=None):
        if _from == None and _to == None:
            return ndarray[(ndarray >= np.min(ndarray)) & (ndarray <= np.max(ndarray))]
        elif _from == None and _to is not None:
            return ndarray[(ndarray >= np.min(ndarray)) & (ndarray <= _to)]
        elif _from is not None and _to == None:
            return ndarray[(ndarray >= _from) & (ndarray <= np.max(ndarray))]
        else:
            return ndarray[(ndarray >= _from) & (ndarray <= _to)]

    def pd_get_between(df, col=col, _from=None, _to=None):
        if _from == None and _to == None:
            return df[
                (df["{}".format(col)] >= df["{}".format(col)].min())
                & (df["{}".format(col)] <= df["{}".format(col)].max())
            ]
        elif _from == None and _to is not None:
            return df[
                (df["{}".format(col)] >= df["{}".format(col)].min())
                & (df["{}".format(col)] <= _to)
            ]
        elif _from is not None and _to == None:
            return df[
                (df["{}".format(col)] >= _from)
                & (df["{}".format(col)] <= df["{}".format(col)].max())
            ]
        else:
            return df[(df["{}".format(col)] >= _from) & (df["{}".format(col)] <= _to)]

    if type(data) == np.ndarray:
        return np_get_between(data, _from, _to)
    elif type(data) == pd.core.frame.DataFrame:
        assert col is not None, "No column has been provided for pandas dataframe"
        return pd_get_between(data, col, _from, _to)


# ------{Slice based on timestamp}------#


def timed_slice(
    data,
    timeseries=None,
    weeks=0,
    days=0,
    hours=0,
    minutes=0,
    seconds=0,
    milliseconds=0,
    microseconds=0,
):
    """
    Get a timed slice from either a  pandas dataframe or numpy array

    Parameters
    ----------

    data : pd.DataFrame|np.array
       A pandas dataframe to slice based on time

    timeseries : str
       Column Name of the timeseries to use

    weeks : int
       How many weeks

    days : int
       How many days

    minutes : int
       how many minutes

    seconds : int
       How many seconds

    milliseconds : int
       How many milliseconds

    microseconds : int
       How many microseconds

    Returns
    -------

   pd.DataFrame|np.array
       Sliced data based on timedeltas
    """

    def np_timed_slice(
        ndarray,
        weeks=0,
        days=0,
        hours=0,
        minutes=0,
        seconds=0,
        milliseconds=0,
        microseconds=0,
    ):
        _weeks = np.timedelta64(weeks, "W")
        _days = np.timedelta64(days, "D")
        _hours = np.timedelta64(hours, "h")
        _minutes = np.timedelta64(minutes, "m")
        _seconds = np.timedelta64(seconds, "s")
        _milliseconds = np.timedelta64(milliseconds, "ms")
        _microseconds = np.timedelta64(microseconds, "us")
        range_max = np.max(data)
        range_min = (
            (((((range_max - _weeks) - _days) - _hours) - _minutes) - _seconds)
            - _milliseconds
        ) - _microseconds
        sliced_df = ndarray[(ndarray >= range_min) & (ndarray <= range_max)]
        return sliced_df

    def pd_timed_slice(
        df,
        timeseries,
        week=0,
        day=0,
        hour=0,
        minute=0,
        second=0,
        millisecond=0,
        microsecond=0,
    ):
        df[timeseries] = pd.to_datetime(df[timeseries])
        range_max = df[timeseries].max()
        range_min = range_max - datetime.timedelta(
            weeks=week,
            days=day,
            hours=hour,
            minutes=minute,
            seconds=second,
            milliseconds=millisecond,
            microseconds=microsecond,
        )
        sliced_df = df[(df[timeseries] >= range_min) & (df[timeseries] <= range_max)]
        return sliced_df

    if type(data) == pd.core.frame.DataFrame:
        assert timeseries is not None, "No column name has been provided for timeseries"
        return pd_timed_slice(
            data,
            timeseries,
            weeks,
            days,
            hours,
            minutes,
            seconds,
            milliseconds,
            microseconds,
        )

    if type(data) == np.ndarray:
        return np_timed_slice(
            data, weeks, days, hours, minutes, seconds, milliseconds, microseconds
        )


# -{inf or NaN to 0}-#


def inf_nan_tozero(data):
    """
    Infinity or NaN to 0

    Parameters
    ----------

    data : pd.DataFrame
       The data that will have inf|nans removed

    Returns
    -------

    pd.DataFrame
        Data with inf and NaNs removed
    """
    data[data == -np.inf] = 0
    data[data == np.inf] = 0
    data[data == np.nan] = 0
    return data


# {Get a masked slice}#


def masked_slice(data, mask, exclude=False):
    """
    Get a slice on a boolean mask

    Parameters
    ----------

    data : np.array
       Array of data to slice

    mask : np.array
       array of booleans to mask

    exclude : bool
       Exclude based on mask or include based on mask

    Returns
    -------

    np.array
       Data array matching the mask
    """
    assert type(data) == np.ndarray, "TypeError: data must be of type np.ndarray!"
    assert type(mask) == np.ndarray, "TypeError: mask must be of type np.ndarray!"
    if exclude:
        return data[~np.in1d(data, mask)]
    else:
        return data[np.in1d(data, mask)]


# {Get subset of data}#


def get_subset(data, col=None, cond_value=0, flag=None):
    """
    Get a subset of the data based on < > != == <= >= flags

    Parameters
    ----------

    data : pd.DataFrame|np.array
       Data to slice

    col : str(Optional)
       column name if using a pd.DataFrame

    cond_value : int|float
       Conditional value to compare against

    flag : str(==,!=,<,<=,>,>=)
       nil

    Returns
    -------

    pd.DataFrame|np.array
        Sliced data based on cond_value and flag
    """

    def pd_get_subset(df, col, cond_value, flag=None):
        if flag == "==":
            return df[df[col] == cond_value]
        elif flag == "<":
            return df[df[col] < cond_value]
        elif flag == "<=":
            return df[df[col] <= cond_value]
        elif flag == ">":
            return df[df[col] > cond_value]
        elif flag == ">=":
            return df[df[col] >= cond_value]
        elif flag == "!=":
            return df[df[col] != cond_value]
        elif flag == None:
            return df

    def np_get_subset(array, cond_value, flag=None):
        if flag == "==":
            return array[array == cond_value]
        elif flag == "<":
            return array[array < cond_value]
        elif flag == "<=":
            return array[array <= cond_value]
        elif flag == ">":
            return array[array > cond_value]
        elif flag == ">=":
            return array[array >= cond_value]
        elif flag == "!=":
            return array[array != cond_value]
        elif flag == None:
            return array

    if type(data) == pd.core.frame.DataFrame:
        assert col is not None, "Data type{} : please provide column name".format(
            type(data)
        )
        return pd_get_subset(data, col, cond_value, flag)
    elif type(data) == np.ndarray:
        return np_get_subset(data, cond_value, flag)
    else:
        raise TypeError(
            "Please provide a Pandas DataFrame or a numpy array, instead of {}".format(
                type(data)
            )
        )


# {Get data with string}#


def pd_string_subset(df, df_col, containing, flag="search"):
    """
    Get a slice based on a string search

    Parameters
    ----------

    df : pd.DataFrame
       Data to slice

    df_col : str
       Column name to take the slice from

    containing : str
       String to make the pattern search with

    flag : str(search|match|extract)
       Pattern search type

    Returns
    -------

    pd.DataFrame
        string subset sliced DataFrame
    """
    if flag == "search":
        return df[df[df_col].str.contains(containing, na=False)]
    elif flag == "match":
        return df[df[df_col].str.match(containing, na=False)]
    elif flag == "extract":
        return df[df[df_col].str.extract(containing, expand=True)]


# ---{Get correlations among columns}---#


def get_top_correlations(df, column, threshold=0.65, top=3):
    column_corr = np.fabs(df.corr()[column].drop(column)).sort_values(
        ascending=False, inplace=False
    )
    top_corr = column_corr[(column_corr > threshold)][:top].index
    correlations = df.corr()[column][top_corr].to_dict()
    return ", ".join(
        "{}: {}".format(col, _percent(val)) for col, val in correlations.items()
    )


def _percent(x):
    x = _number_format(100 * x)
    return "{}%".format(x)


def _number_format(x):
    eps = 1e-9
    num_format = "{0:,.0f}" if abs(int(x) - x) < eps else "{0:,.2f}"
    return num_format.format(x)


def get_uniques(df):
    return pd.Series(dict((c, df[c].nunique()) for c in df.columns), name="uniques")


def list_cols_to_pd(input, columns):
    return pd.DataFrame(dict(zip(columns, input)), columns=columns, index=[0])


def get_columns(df, usage="include", columns=None):
    columns_excluded = pd.Index([])
    columns_included = df.columns
    if usage == "include":
        try:
            columns_included = columns_included.intersection(pd.Index(columns))
        except TypeError:
            pass
    elif usage == "exclude":
        try:
            columns_excluded = columns_excluded.union(pd.Index(columns))
        except TypeError:
            pass
    columns_included = columns_included.difference(columns_excluded)
    return columns_included.intersection(df.columns)


def get_date_summary(df, column):
    series = df[column]
    stats = {"min": series.min(), "max": series.max()}
    stats["range"] = stats["max"] - stats["min"]
    return stats
    # return pd.concat([pd.Series(stats, name=column), df.column_stats.ix[:,column]])


def categorical_change(df, col):
    cats = np.array([-1, 0, 1])  # decrease, --- , increase
    change = cats[
        np.sign(np.append(0, np.diff(df["{}".format(col)].values, 1))).astype(np.uint8)
        + 1
    ]
    df["chn_{}".format(col)] = change
    return df


def cummulative_change(df, col):
    change = (
        df["{}".format(col)]
        .groupby((df["{}".format(col)] == df["{}".format(col)].shift()).cumsum())
        .cumcount()
    )
    return change


def get_xy(df, target_col):
    X = df[df.columns.difference([target_col])]
    Y = df[[target_col]]
    return X, Y


def df_index_contains(df, df_2, exclude=False):
    """
    * type-def ::pd.df :: pd.df -> pd.df
    * ---------------{Function}---------------
    * See if values in index match from one dataframe to another . . .
    * ----------------{Params}----------------
    * : pandas Dataframe to check
    * : pandas dataFrame to check against
    * : Whether to exclude or include if contained
    * ----------------{Returns}---------------
    * df with values contained/not contained in index of df2 . . .
    """
    if exclude:
        return df[~df.index.isin(df_2.index)].reset_index()
    else:
        return df[df.index.isin(df_2.index)].reset_index()


def isnan(x):
    if isinstance(x, (int, float, complex)) and math.isnan(x):
        return True


def fill_na_elist(df):
    df.apply(lambda x: x.apply(lambda x: [] if isnan(x) else x))


# --{data chunking}--#


def data_chunker(data, chunks=None, max_len=None):
    if chunks is not None and max_len is None:
        chunk_size = ceil(data.shape[0] / chunks)
        val = 0
        for chunk_num in range(chunks):
            if chunk_num == 0:
                yield data[val:chunk_size]
            val += chunk_size
            yield data[val : (val + chunk_size)]

    elif max_len is not None and chunks is None:
        val = 0
        while True:
            if val == 0:
                val += max_len
                yield data[0:(val)]
            val += max_len
            yield data[(val - max_len) : val]


def sliding_window(data, segment_length, slide_length, flag="chunks"):
    def iter_sliding_window(data, segment_length, slide_length):
        for start_position in range(0, len(data), slide_length):
            end_position = start_position + segment_length
            yield data[start_position:end_position]

    def bulk_sliding_window(data, segment_length, slide_length):
        segments = []
        for start_position in range(0, len(data), slide_length):
            end_position = start_position + segment_length
            # make a copy so changes to 'segments doesn't modify original data
            segment = np.copy(data[start_position:end_position])
            # if we're at the end and we've got a truncated segment, drop it
            if len(segment) != segment_length:
                continue
            segments.append(segment)
        print("Produced {} waveform segments".format(len(segments)))
        return segments

    if flag == "chunks":
        return bulk_sliding_window(data, segment_length, slide_length)
    elif flag == "lazy":
        return iter_sliding_window(data, segment_length, slide_length)


def pad(A, npads):
    _npads = npads - len(A)
    return np.pad(A, pad_width=_npads, mode="constant", constant_values=0)[_npads:]


def pad_floats(A, length):
    arr = np.zeros(length)
    arr[: len(A)] = A
    return arr


def duplicated_varnames(df):
    """Return a dict of all variable names that
    are duplicated in a given dataframe."""
    repeat_dict = {}
    var_list = list(df)  # list of varnames as strings
    for varname in var_list:
        # make a list of all instances of that varname
        test_list = [v for v in var_list if v == varname]
        # if more than one instance, report duplications in repeat_dict
        if len(test_list) > 1:
            repeat_dict[varname] = len(test_list)
    return repeat_dict


def vectorize_1D(func, np_array):
    np_array = np_array.reshape(np.array.shape[0], 1)
    result = np.apply_along_axis(func, 1, np_array)
    return result


# ------------{SQL utilities}-----------#


def insert_df(df, *args, **kwargs):
    """
    * -------------{Function}---------------
    * insert pandas dataframe using multiple threads . . .
    * -------------{params}-----------------
    * : df
    * : 'table_name'
    * : connection engine
    * : if_exists='append'
    * -------------{extra}------------------
    * only works with non-locking databases if a sqlite database is used
    * it will yield a ProgrammingError
    """
    nworkers = 4

    chunksize = floor(df.shape[0] / nworkers)
    chunks = [(chunksize * i, (chunksize * i) + chunksize) for i in range(nworkers)]
    chunks.append((chunksize * nworkers, df.shape[0]))
    pool = ThreadPool(nworkers)

    def worker(chunk):
        i, j = chunk
        df.iloc[i:j, :].to_sql(*args, **kwargs)

    pool.map(worker, chunks)
    pool.close()
    pool.join()


# TODO: PARALLELIZE THIS
def tosql(df, *args, **kargs):
    CHUNKSIZE = 10000
    INITIAL_CHUNK = 100
    if len(df) > CHUNKSIZE:
        df.iloc[:INITIAL_CHUNK, :].to_sql(*args, **kargs)
    if kargs["if_exists"] == "replace":
        kargs["if_exists"] = "append"
    workers = []
    for i in range((len(df) - INITIAL_CHUNK) // CHUNKSIZE):
        t = threading.Thread(
            target=lambda: df.iloc[
                INITIAL_CHUNK + i * CHUNKSIZE : INITIAL_CHUNK + (i + 1) * CHUNKSIZE, :
            ].to_sql(*args, **kargs)
        )
        t.start()
        workers.append(t)
        df.iloc[INITIAL_CHUNK + (i + 1) * CHUNKSIZE :, :].to_sql(*args, **kargs)
    [t.join() for t in workers]


# BY FAR FASTEST
def to_sql(engine, df, table, if_exists="fail", sep="\t", encoding="utf8"):
    """
   * -------------{Function}---------------
   * write_to_sql_efficiently . . .
   * -------------{params}-----------------
   * : connection engine
   * : pandas dataframe
   * : 'table_name'
   * :
   * -------------{extra}------------------
   * at the moment would only work with PostgreSQL due to copy_from call . . .
   """
    # Create Table
    df[:0].to_sql(table, engine, if_exists=if_exists)

    # Prepare data
    output = StringIO()
    df.to_csv(output, sep=sep, header=False, encoding=encoding)
    output.seek(0)

    # Insert data
    try:
        connection = engine.raw_connection()
        cursor = connection.cursor()
    except Exception:
        cursor = engine.cursor()
    cursor.copy_from(output, table, sep=sep, null="")
    connection.commit()
    cursor.close()


def process_flags_pd(df, value, column):
    return df[df[column] & value != 0]


def fast_np_fillna(a):
    # Eliminate NaN
    ind = np.where(~pd.isnull(a))[0]
    first, last = ind[0], ind[-1]
    a[:first] = a[first]
    a[last + 1 :] = a[last]
    return a


def fast_np_fillna(a):
    # Eliminate None
    ind = np.where(~np.equal(a, None))[0]
    first, last = ind[0], ind[-1]
    a[:first] = a[first]
    a[last + 1 :] = a[last]
    return a


def not_none(data):
    if data == "none":
        return 0
    else:
        return 1


def greater_than_zero(data):
    if data > 0:
        return 1
    else:
        return 0


def sql_select_chunker(
    engine, cols="*", table="ticks", optionals="", order_by="timestamp", size=None
):
    if size is None:
        size = table_size(engine, table)
    if size > 1e9:
        chunks = 100000
    elif size > 1e6:
        chunks = 1000
    else:
        chunks = 10
    chunksize = ceil(size / chunks)
    offset = 0
    for i in range(chunksize):
        query = """select {cols}
        from {table}
        {optionals}
        ORDER BY {order_by}
        LIMIT {chunksize}
        OFFSET {offset}""".format(
            cols=cols,
            table=table,
            optionals=optionals,
            order_by=order_by,
            chunksize=chunksize,
            offset=offset,
        )

        if offset <= size:
            offset += chunksize
            yield query
        else:
            print("done")
            break


def reshape_multi_dimensional(data):
    return data.reshape(1, data.shape[0])


def force_numeric_sort(df):
    if df.empty:
        return df
    df.LAG = pd.to_numeric(df.LAG, errors="coerce")

    df = df.sort_values(by=["LAG"], ascending=False)
    return df


def ndistinct(x):
    out = len(np.unique(x))
    print("There are", out, "distinct values.")


def incremental_bounds_df(size, increments):
    from math import ceil

    lb = 0
    ub = increments
    its = ceil(size / increments)
    return (lb, ub, its)


def pd_apply_debug(func_to_apply, df, col_name, increments=1000):
    """
    Utility fuction to debug pandas.apply calls and retrieve the rows yielding the error

    Parameters
    ----------

    func_to_apply : func
       A function to apply to each row

    df : pd.DataFrame
       Pandas dataframe to apply function to

    col_name : str
       Name of the column

    increments : int
       stride length also df size on return

    Returns
    -------

    pd.DataFrame
         A pandas dataframe of size={increments} containing the rows that yielded an error

    Raises
    ------

    Exception
        The exception that led to the function failing application[printed as a side effect]
    """
    from tqdm import tqdm

    tqdm.pandas()
    lb, ub, its = incremental_bounds_df(df.shape[0], increments)
    for i in range(its):
        sample = df[lb:ub]
        lb += increments
        ub += increments
        try:
            sample[col_name].progress_apply(func_to_apply)
        except Exception as e:
            print(e)
            return sample


def pd_to_klepto_stream(df, path, increments=1000):
    import gc
    import klepto

    lb, ub, its = incremental_bounds_df(df.shape[0], increments)
    d = klepto.archives.dir_archive(path, cached=True, serialized=True)
    df["_idx_"] = df.index
    for i in range(its):
        sample = df[lb:ub]
        d[i] = sample
        d.dump()
        d.clear()
        gc.collect()
        lb += increments
        ub += increments


def pd_from_klepto_stream(path):
    import gc
    import klepto

    d = klepto.archives.dir_archive(path, cached=True, serialized=True)
    slices = len(d.__dict__["__archive__"])
    for i in range(slices):
        if i == 0:
            df = d.__dict__["__archive__"][i]
        else:
            temp = d.__dict__["__archive__"][i]
            df = pd.concat((temp, df))
            del temp
            gc.collect()
    df.set_index("_idx_", inplace=True, drop=True)
    df.sort_index(inplace=True)
    return df


def pd_stream_apply(df, col_name, func, increments=1000, progress=False, keep="False"):
    import gc
    import klepto
    from hashlib import sha256
    from os import remove
    import shutil

    if progress:
        from tqdm import tqdm

        tqdm.pandas()

    lb, ub, its = incremental_bounds_df(df.shape[0], increments)
    path = sha256(str(increments).encode()).hexdigest()

    df["_idx_"] = df.index

    d = klepto.archives.dir_archive(path, cached=True, serialized=True)
    for i in range(its):
        sample = df[lb:ub]

        if progress:
            sample[col_name] = sample[col_name].progress_apply(func)
        else:
            sample[col_name] = sample[col_name].apply(func)
        d[i] = sample
        d.dump()
        d.clear()
        gc.collect()
        lb += increments
        ub += increments

    df = pd_from_klepto_stream(path)
    if keep == False:
        shutil.rmtree(path)
    return df


def pd_stream_parallel_apply(
    df, col_name, func, increments=1000, keep="False", pool_size=4
):
    import gc
    import klepto
    import shutil
    import multiprocessing
    from hashlib import sha256
    from os import remove

    pool = multiprocessing.Pool(multiprocessing.cpu_count())

    lb, ub, its = incremental_bounds_df(df.shape[0], increments)
    path = sha256(str(increments).encode()).hexdigest()

    df["_idx_"] = df.index

    d = klepto.archives.dir_archive(path, cached=True, serialized=True)
    for i in range(its):
        sample = df[lb:ub]

        results = pool.map(func, list(sample[col_name]))

        sample[col_name] = results
        d[i] = sample
        d.dump()
        d.clear()
        gc.collect()
        lb += increments
        ub += increments

    pool.close()
    pool.join()

    df = pd_from_klepto_stream(path)
    if keep == False:
        shutil.rmtree(path)
    return df


def pd_parallel_apply(data, col_name, func, num_partitions=10):
    # WARNING: Only works currently with non-nested data in a pd.series
    # FROM:http://blog.adeel.io/2016/11/06/parallelize-pandas-map-or-apply/
    import multiprocessing

    data_split = np.array_split(data[col_name], num_partitions)
    pool = multiprocessing.Pool(multiprocessing.cpu_count())
    data[col_name] = pd.concat(pool.map(func, data_split))
    pool.close()
    pool.join()
    return data


def np_parallel_apply_along(data, function, axis=1, parts=4, threads=4):
    # WARNING TODO: use ast to use only the given parts/threads

    split_array = np.array_split(data, parts)
    pool = Pool(threads)
    ast_string = []
    ast_final = ""
    for i in range(parts):
        temp = "(function, {}, split_array[{}].reshape((split_array[{}].shape[0], 1)))".format(
            axis, i, i
        )
        ast_string.append(temp)
        ast_final += "eval(ast_string[{}]),".format(i)

    result = pool.starmap(
        np.apply_along_axis,
        [
            eval(ast_string[0]),
            eval(ast_string[1]),
            eval(ast_string[2]),
            eval(ast_string[3]),
        ],
    )
    result = np.concatenate(([*result]), axis=0)
    return result


def split_dataframe(df, sections=10):
    from math import ceil

    part_size = ceil(df_u.shape[0] // sections)
    result = []
    lb = 0
    ub = part_size
    for i in range(sections):
        temp_df = df[lb:ub].copy()
        result.append(temp_df)
        lb = lb + part_size
        ub = ub + part_size
    return result


def fill_na(df):
    for col in df:
        # get dtype for column
        dt = df[col].dtype
        # check if it is a number
        if dt == int or dt == float:
            df[col].fillna(0, inplace=True)
        elif dt == np.dtype("<M8[ns]"):
            df[col].fillna(datetime.datetime(2000, 1, 1, 0, 0, 0), inplace=True)
        else:
            df[col].fillna("", inplace=True)
    return df


def inner_join(con, table1, table2, col):
    a = pd.read_sql("select * from {} limit 1;".format(table1), con).keys()
    b = pd.read_sql("select * from {} limit 1;".format(table2), con).keys()
    intersection = [table1 + "." + i for i in a.intersection(b)]
    intersection.extend(
        [table2 + "." + i + " as " + table2 + "_" + i for i in a.intersection(b)]
    )
    union = list(a.symmetric_difference(b))
    union.extend(intersection)
    cols = ", ".join(union)
    result = "select {} from {} INNER JOIN {} ON {}.{}={}.{}".format(
        cols, table1, table2, table1, col, table2, col
    )
    return result + " "


def nest_inner(con, query, view1, table2, col, col2=None):
    a = pd.read_sql(query + " limit 1", con).keys()
    b = pd.read_sql("select * from {} limit 1;".format(table2), con).keys()
    intersection = [view1 + "." + i for i in a.intersection(b)]
    intersection.extend(
        [table2 + "." + i + " as " + table2 + "_" + i for i in a.intersection(b)]
    )
    union = list(a.symmetric_difference(b))
    union.extend(intersection)
    cols = ", ".join(union)
    if col2:
        result = "select {} from ({}) as {} INNER JOIN {} ON {}.{}={}.{}".format(
            cols, query, view1, table2, view1, col, table2, col2
        )
    else:
        result = "select {} from ({}) as {} INNER JOIN {} ON {}.{}={}.{}".format(
            cols, query, view1, table2, view1, col, table2, col
        )
    return result + " "


def fillna_sampled(x, reproducibility_seed=0):
    """
    fill NA with a random sample from existing series

    Parameters
    ----------
    x: pd.Series
       A Pandas Series containing the values to sample from
       after non-destructively dropping NaNs
    reproducibility_seed: int
       Random Seed, used to seed the random number generator,
       will allow the data to be reproducible
    Return
    ------
    filled : pd.DataFrame, pd.Series
        Imputed pandas series through sampling on existing values.
    """
    n_nans = len(x[pd.isnull(x)])
    filled = x.fillna(
        pd.Series(
            x.dropna(inplace=False).sample(n=n_nans, random_state=reproducibility_seed)
        )
    )

    return filled


def iter_range_pd(df):
    for i in range(df.shape[0]):
        yield (df.iloc[i])


def pd_csv_to_buffer(data):
    import io

    u = data.to_csv().encode("latin-1")
    out_buffer = io.BytesIO()
    out_buffer.seek(0)
    out_buffer.write(u)
    out_buffer.seek(0)
    return out_buffer


def transform_aos_soa(dict_list):
    cols = list(dict_list[0].keys())
    soa = {}
    for i in cols:
        soa[i] = []
        for j in range(len(dict_list)):
            try:
                soa[i].append(dict_list[j][i])
            except Exception as e:
                soa[i].append(None)
    return soa


def transform_aos_pd(dict_list):
    """
    Load a list of dicts into pandas, very efficiently. Great for dealing with rest.json results

    Parameters
    ----------

    dict_list : List
       A list of dicts [{}]

    Returns
    -------

    pd.DataFrame
        A pandas dataframe with all the data from the dicts
    """
    return pd.DataFrame(transform_aos_soa(dict_list))


def serialize_pd_str(df):
    import base64
    import io

    temp_io = io.BytesIO()
    df.to_pickle(temp_io, compression=None)
    temp_io.seek(0)
    result = base64.b64encode(temp_io.read())
    return result.decode()


def deserialize_pd_str(data):
    import base64
    import io

    temp_io = io.BytesIO()
    temp_io.write(base64.b64decode(data))
    temp_io.seek(0)
    result = pd.read_pickle(temp_io)
    return result


def serialize_pd_csv(df):
    import base64
    import io

    temp_io = io.StringIO()
    df.to_csv(temp_io)
    temp_io.seek(0)
    result = base64.b64encode(temp_io.read().encode())
    return result.decode()


def deserialize_pd_csv(data):
    import base64
    import io

    temp_io = io.BytesIO()
    temp_io.write(base64.b64decode(data))
    temp_io.seek(0)
    result = pd.read_csv(temp_io)
    return result


def fill_sequential(df):
    if pd_get_nan(df).shape[0] == 0:
        return df

    uu = pd_get_nan(df)
    fill = pd_get_not_nan(df)

    for index, i in enumerate(fill.values):
        df.iloc[uu[index]] = i
    fill_sequential(df)


def pd_get_nan(df):
    if type(df) == pd.core.series.Series:
        return df[df.isna()]
    else:
        return df[df.isna()[df.columns[0]]]


def pd_get_not_nan(df):
    if type(df) == pd.core.series.Series:
        return df[~df.isna()]
    else:
        return df[~df.isna()[df.columns[0]]]


def broadcast_fill(df_or_series, nan_series, fill_array):
    """
    Broadcast an array into a sequence of indexes

    Parameters
    ----------
    df_or_series : pd.DataFrame | pd.core.series.Series
        A df or series to modify

    nan_series : pd.core.series.Series
        A series of NaNs containing the index of NaNs in df_or_series

    fill_array : np.array
       Array to fill df_or_series with

    Returns
    -------
    pd.core.series.Series:
       Series with NaNs filled
    """
    nan_num = nan_series.shape[0]
    if type(df_or_series) == pd.core.series.Series:
        result = df_or_series.loc[list(nan_series.index)] = fill_array[:nan_num]
        return result
    else:
        result = df_or_series.iloc[:, 0].loc[list(nan_series.index)] = fill_array[
            :nan_num
        ]
        return result


def pd_split_str(series, sep):
    a, b = series.str.split(sep, 1).str
    return a, b


def pd_get_dummies_concat(source_df, column):
    """
    One hot encode a column and concatenate horizontally [] + []

    Parameters
    ----------

    source_df : pd.DataFrame
       Source dataframe to concatenate into

    column : string
       column to one-hot-encode

    Returns
    -------

    pd.DataFrame:
        One hot encoded dataframe concatenated into source_df
    """
    return pd.concat([source_df, pd.get_dummies(df[column])], axis=1, sort=False)


def pd_histogram(series: pd.Series):
    """
    Get the histogram data as from a pd.Series

    Parameters
    ----------

    series : pd.Series
       A pandas series to extract the value_counts

    Returns
    -------

    pd.Series
        A pandas series with the resulting histogram data
    """
    counts, bins = np.histogram(series)
    return pd.Series(counts, index=bins[:-1])


def pd_plot_hist(series: pd.Series):
    """
    Plot the histogram from a pandas series

    Parameters
    ----------

    series : pd.Series
       A pandas series to extract histogram data from

    Returns
    -------

    matplotlib.axes
        Axes data from matplotlib with the histogram plot
    """
    return pd_histogram(series).plot.bar()


def topackedbits(x):
    return np.unpackbits(np.frombuffer(np.asarray(x), dtype=np.uint8))


def frompackedbits(bits, dtype=np.int64):
    return np.frombuffer(np.packbits(bits), dtype=dtype)


def rows_with_nan(df, col=None):
    if col != None:
        temp_df = df[col]
    else:
        temp_df = df

    if type(temp_df) == pd.core.series.Series:
        is_NaN = temp_df.isnull()
        row_has_NaN = pd.DataFrame(is_NaN).any(axis=1)
        rows_with_NaN = df[row_has_NaN]
        return rows_with_NaN
    elif type(temp_df) == pd.core.frame.DataFrame:
        is_NaN = temp_df.isnull()
        row_has_NaN = is_NaN.any(axis=1)
        rows_with_NaN = temp_df[row_has_NaN]
        return rows_with_NaN


def drop_contains_pd(df, col, contained_string):
    df.drop(
        df[col][df[col].str.contains(contained_string)].index, inplace=True,
    )
    return df


def partition_df(df, col):
    for current_element in df[col].unique():
        result = df[df[col] == current_element]
        yield result
