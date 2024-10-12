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


def get_non_matching_columns(data, substrings):
    if isinstance(data, pd.DataFrame):
        columns = data.columns
    elif isinstance(data, list):
        columns = data
    else:
        raise ValueError("Input must be a pandas DataFrame or a list of column names.")

    if isinstance(substrings, str):
        substrings = [substrings]
    elif not isinstance(substrings, list):
        raise ValueError("Substrings must be a string or a list of strings.")

    non_matching_columns = [
        col for col in columns if not any(sub in col for sub in substrings)
    ]

    return non_matching_columns


def pprint_df(data, float_format="{:.4f}", header_style="bold magenta"):
    """
    Enhanced pretty print for pandas DataFrame, DataFrame columns (Pandas Index), or lists,
    using the rich library for improved formatting and flexibility.

    :param data: The data to print. Can be a pandas DataFrame, DataFrame columns, or a list.
    :param float_format: The format string for float values. Defaults to "{:.4f}".
    :param header_style: The style of the table header. Defaults to "bold magenta".
    """
    from rich.console import Console
    from rich.table import Table

    console = Console()
    table = Table(show_header=True, header_style=header_style)

    if isinstance(data, pd.DataFrame):
        # Add columns to the table for DataFrame
        for column in data.columns:
            table.add_column(column)
        # Format float columns and convert all values to strings for DataFrame
        for _, row in data.iterrows():
            formatted_row = [
                float_format.format(val) if isinstance(val, float) else str(val)
                for val in row
            ]
            table.add_row(*formatted_row)
    elif isinstance(data, pd.Index) or isinstance(data, list):
        # Handle a single row of data for Pandas Index or list
        table.add_column("Values")
        for item in data:
            # Here we assume the items are not float so direct conversion to string
            table.add_row(str(item))
    else:
        # Handle unsupported types
        console.print(
            "[bold red]Unsupported data type. Please provide a DataFrame, a DataFrame columns (Index), or a list.[/bold red]"
        )
        return
    # Print the table
    console.print(table)


def identify_unhashable_columns(df: pd.DataFrame):
    """
    * ---------------Function---------------
    * Identifies DataFrame columns that contain unhashable types such as lists.
    * ----------------Returns---------------
    * -> list[str] : A list of column names that contain unhashable types
    * ----------------Params----------------
    * df :: pd.DataFrame : The input DataFrame to check for unhashable columns
    * ----------------Usage-----------------
    * This function can be used to identify columns in a DataFrame that contain unhashable types,
    such as lists. This is useful when working with data structures that require hashable types.
    * ----------------Notes-----------------
    * This function iterates over each column in the input DataFrame and checks if any value in the column is a list.
    If a column contains at least one list, it is added to the list of unhashable columns.
    """
    unhashable_columns = []

    for column in df.columns:
        if df[column].apply(lambda x: isinstance(x, list)).any():
            unhashable_columns.append(column)

    return unhashable_columns


def dump_string_to_file(s: str, file_path: str = None):
    """
    Dumps a given string into a file. If no file path is provided, it defaults to the current working directory.

    :param s: String to be written to the file.
    :param file_path: Optional; Path of the file where the string will be dumped. Defaults to 'temp.txt' in the current working directory.
    """
    if file_path is None:
        file_path = os.path.join(os.getcwd(), "temp.txt")

    with open(file_path, "w") as file:
        file.write(s)
    print(f"String successfully written to {file_path}")


def reveal_incomplete_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    *---------------Function---------------
    *Return a DataFrame showing only rows with at least one NaN,
    but only in columns that also contain at least one NaN.
    *----------------Returns---------------
    *-> pd.DataFrame
    *----------------Params----------------
    *df :: pd.DataFrame
    *----------------Usage----------------
    *Use this function to identify and extract rows with missing values in a pandas DataFrame.
    *Example: reveal_incomplete_rows(my_df) would return a new DataFrame with only the rows that have NaN values in columns that also have NaN values.
    *----------------Notes-----------------
    *This function is useful for data cleaning and exploratory data analysis.
    *Be cautious when using this function with large datasets, as it may return a large number of rows.
    """
    # Find columns with at least one NaN
    cols_with_nan = df.columns[df.isna().any()]

    # Filter rows that have at least one NaN anywhere
    rows_with_nan = df[df.isna().any(axis=1)]

    return rows_with_nan[cols_with_nan]


def pd_batch_iterator(df):
    batch_size = 20
    num_rows = len(df)
    num_batches = (num_rows // batch_size) + (num_rows % batch_size > 0)
    current_batch = 0

    while current_batch < num_batches:
        start_idx = current_batch * batch_size
        end_idx = min(start_idx + batch_size, num_rows)
        yield df.iloc[start_idx:end_idx]
        current_batch += 1


def pretty_print_dataframe_samples(df, string_to_find=None):
    """
    Prints each column name with its corresponding sample value in a dataframe.
    Column names are printed in yellow, and the sample values are printed in grey.
    Prints column names containing a specified string in red.

    Parameters:
    - df: pandas DataFrame
    - string_to_find: String to be searched in column names (default: None)

    Returns:
    - None
    """
    import pandas as pd
    from termcolor import colored

    print("Column Name : Sample")
    for column in df.columns:
        sample_value = df[column].iloc[0]
        if string_to_find and string_to_find in column:
            formatted_output = (
                f"{colored(column, 'red')} : {colored(sample_value, 'grey')}"
            )
        else:
            formatted_output = (
                f"{colored(column, 'yellow')} : {colored(sample_value, 'grey')}"
            )
        print(formatted_output)


def included_cols_excluded_cols(df, excluded_columns):
    return list(set(df.columns) - set(excluded_columns))


def get_sample_rows_with_na(df, n=3):
    # Get a list of columns that have missing values
    cols_with_na = df.columns[df.isna().any()].tolist()

    # Create a dictionary to store sample rows
    sample_rows = {}

    for col in cols_with_na:
        # Get a random sample of a row with a missing value in the column
        sample_rows[col] = df[df[col].isna()].sample(n).to_dict("records")

    return sample_rows


def sample_nan_rows(df):
    """Return a single sample row that has a NaN value for each column"""
    nan_samples = {}
    for column in df.columns:
        nan_rows = df[df[column].isna()]
        if not nan_rows.empty:
            nan_samples[column] = nan_rows.sample(1)
    return nan_samples


def pd_dump_to_string_file(df):
    from tabulate import tabulate

    dump_string_to_file(tabulate(df[:2], headers="keys", tablefmt="psql"))
    return 1


def show_matching_columns(dataframes):
    common_columns = set(dataframes[0].columns.str.lower())
    all_columns = set(dataframes[0].columns.str.lower())
    column_types = {}

    for i, df in enumerate(dataframes[1:], 1):
        common_columns &= set(df.columns.str.lower())
        all_columns |= set(df.columns.str.lower())

    diff_columns = all_columns - common_columns

    # Prepare column_types dictionary
    for col in common_columns:
        column_types[col] = [
            df[df.columns[df.columns.str.lower() == col].tolist()[0]].dtype
            for df in dataframes
            if col in df.columns.str.lower().tolist()
        ]

    # ANSI escape codes for colors
    green = "\033[92m"
    red = "\033[91m"
    yellow = "\033[93m"
    end = "\033[0m"
    color_idx = ["\033[9" + str(i % 10) + "m" for i in range(len(dataframes))]

    # Print common columns
    print(green + "Common Columns: " + ", ".join(common_columns) + end)

    # Print different columns with DataFrame index color matching
    print("Different Columns:")
    for col in diff_columns:
        color_strs = []
        for i, df in enumerate(dataframes):
            if col in df.columns.str.lower():
                color_strs.append(color_idx[i] + str(i) + end)
        print(red + col + " (" + ", ".join(color_strs) + ")" + end)

    # Print columns with differing types
    print("Different Types:")
    for col, types in column_types.items():
        if (
            len(set(types)) > 1
        ):  # If there's more than one unique type across DataFrames
            type_strs = []
            for i, df in enumerate(dataframes):
                if col in df.columns.str.lower():
                    type_strs.append(color_idx[i] + str(df[col].dtype) + end)
            print(yellow + col + " (" + ", ".join(type_strs) + ")" + end)


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


def pd_load_tuple_py(tuple_list):
    """
    * type-def ::(List[Tuple[str, Any]]) -> pd.DataFrame
    * ---------------{Function}---------------
        * Converts a list of tuples into a pandas DataFrame, where the first element of each tuple becomes the column name.
    * ----------------{Returns}---------------
        * : df ::pd.DataFrame | A DataFrame with column names from the first element of each tuple in the input list
    * ----------------{Params}----------------
        * : tuple_list ::List[Tuple[str, Any]] | A list of tuples to be converted into a DataFrame
    * ----------------{Usage}-----------------
        * >>> tuple_list = [("A", 1, 2), ("B", 3, 4), ("C", 5, 6)]
        * >>> df = pd_load_tuple(tuple_list)
        * >>> print(df)
    * ----------------{Output}----------------
        *    A  B  C
        * 1  1  3  5
        * 2  2  4  6
    * ----------------{Notes}-----------------
        * This function is useful for converting a list of tuples into a pandas DataFrame with specified column names.
    """
    df = pd.DataFrame(tuple_list).T
    df.columns = df.iloc[0]
    df = df.reindex(df.index.drop(0))
    return df


def pd_load_tuple_np(tuple_list):
    """
    * type-def ::(List[Tuple[str, Any]]) -> pd.DataFrame
    * ---------------{Function}---------------
        * Converts a list of tuples into a pandas DataFrame, where the first element of each tuple becomes the column name.
    * ----------------{Returns}---------------
        * : df ::pd.DataFrame | A DataFrame with column names from the first element of each tuple in the input list
    * ----------------{Params}----------------
        * : tuple_list ::List[Tuple[str, Any]] | A list of tuples to be converted into a DataFrame
    * ----------------{Usage}-----------------
        * >>> tuple_list = [("A", 1, 2), ("B", 3, 4), ("C", 5, 6)]
        * >>> df = pd_load_tuple(tuple_list)
        * >>> print(df)
    * ----------------{Output}----------------
        *    index  1  2
        * 0      A  1  2
        * 1      B  3  4
        * 2      C  5  6
    * ----------------{Notes}-----------------
        * This function is useful for converting a list of tuples into a pandas DataFrame with specified column names.
    """
    tuple_array = np.array(tuple_list)
    tuple_array_transposed = tuple_array.T
    df = pd.DataFrame(tuple_array_transposed[:, 1:], index=tuple_array_transposed[:, 0])
    df.reset_index(inplace=True)
    return df


def pd_concat_list_dict(list_dict):
    """
    * type-def ::(List[Dict[str, Any]]) -> pd.DataFrame
    * ---------------{Function}---------------
        * Concatenates a list of dictionaries into a pandas DataFrame.
    * ----------------{Returns}---------------
        * : df ::pd.DataFrame | A concatenated DataFrame built from the input list of dictionaries
    * ----------------{Params}----------------
        * : list_dict ::List[Dict[str, Any]] | A list of dictionaries to be concatenated into a DataFrame
    * ----------------{Usage}-----------------
        * >>> list_dict = [{"A": 1, "B": 2}, {"A": 3, "B": 4}, {"A": 5, "B": 6}]
        * >>> df = pd_concat_list_dict(list_dict)
        * >>> print(df)
    * ----------------{Output}----------------
        *    index  1
        * 0      A  1
        * 1      B  2
        * 0      A  3
        * 1      B  4
        * 0      A  5
        * 1      B  6
    * ----------------{Notes}-----------------
        * This function is useful for concatenating a list of dictionaries into a single pandas DataFrame.
    """
    for i in range(len(list_dict)):
        if i == 0:
            df = pd_load_tuple(list(list_dict[i].items()))
        else:
            df = df.append(pd_load_tuple(list(list_dict[i].items())))
    return df


def pd_csv_to_io(df):
    """
    * type-def ::(pd.DataFrame) -> io.BytesIO
    * ---------------{Function}---------------
        * Converts a pandas DataFrame to a CSV format in an in-memory binary buffer.
    * ----------------{Returns}---------------
        * : buffer ::io.BytesIO | A binary buffer containing the CSV representation of the input DataFrame
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | A DataFrame to be converted to CSV format
    * ----------------{Usage}-----------------
        * >>> import pandas as pd
        * >>> data = {"A": [1, 2, 3], "B": [4, 5, 6]}
        * >>> df = pd.DataFrame(data)
        * >>> buffer = pd_csv_to_io(df)
        * >>> print(buffer.getvalue().decode())
    * ----------------{Output}----------------
        * ,A,B
        * 0,1,4
        * 1,2,5
        * 2,3,6
        *
    * ----------------{Notes}-----------------
        * This function is useful for converting a DataFrame to an in-memory binary buffer, which can then be used to store or send the CSV data without writing to disk.
    """
    import io

    buffer = io.BytesIO()
    df.to_csv(buffer)
    buffer.seek(0)
    return buffer


def chunk_select_mysql(con, table_name, order_by, stride_length=1000):
    """
    * type-def ::(Connection, str, str, int) -> Generator[pd.DataFrame, None, None]
    * ---------------{Function}---------------
        * Yields chunks of data from a MySQL table, one chunk at a time.
    * ----------------{Returns}---------------
        * : generator ::Generator[pd.DataFrame, None, None] | A generator yielding DataFrames containing chunks of the table data
    * ----------------{Params}----------------
        * : con ::Connection | A connection object to the MySQL database
        * : table_name ::str | The name of the table to fetch data from
        * : order_by ::str | The column to order the data by when fetching chunks
        * : stride_length ::int | The number of rows in each chunk (default: 1000)
    * ----------------{Usage}-----------------
        * >>> import mysql.connector as mariadb
        * >>> mariadb_connection = mariadb.connect(user='username', password='password', database='mydb')
        * >>> table_name = 'mytable'
        * >>> order_by = 'id'
        * >>> stride_length = 1000
        * >>> for chunk in chunk_select_mysql(mariadb_connection, table_name, order_by, stride_length):
        * ...     print(chunk)
    * ----------------{Notes}-----------------
        * This function is useful for fetching large tables in chunks, reducing memory usage and enabling more efficient processing.
    """
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
    """
    * type-def ::(int, int) -> int
    * ---------------{Function}---------------
        * Calculate the stride length needed to divide a dataset into a specified number of chunks.
    * ----------------{Returns}---------------
        * : stride ::int | The calculated stride length
    * ----------------{Params}----------------
        * : size_data ::int | The total size of the dataset
        * : chunks ::int | The number of chunks to divide the dataset into
    * ----------------{Usage}-----------------
        * >>> size_data = 5000
        * >>> chunks = 10
        * >>> stride_length = get_stride_len(size_data, chunks)
        * >>> print(stride_length)
    * ----------------{Output}----------------
        * 500
    * ----------------{Notes}-----------------
        * This function is useful for determining the stride length needed to divide a dataset into a specified number of chunks.
    """
    from math import ceil

    stride = ceil(size_data / chunks)
    return stride


# {Get attributes pandas}#


def get_attributes(mod):
    """
    * type-def ::(Module) -> List[Tuple[str, Any]]
    * ---------------{Function}---------------
        * Get a list of attributes for a given module, excluding private and special attributes.
    * ----------------{Returns}---------------
        * : attributes ::List[Tuple[str, Any]] | A list of tuples containing attribute names and their values
    * ----------------{Params}----------------
        * : mod ::Module | The module to get attributes from
    * ----------------{Usage}-----------------
        * >>> import numpy as np
        * >>> attributes = get_attributes(np)
        * >>> print(attributes)
    * ----------------{Output}----------------
        * [('ALLOW_THREADS', 1), ('BUFSIZE', 8192), ('CLIP', 0), ...]
    * ----------------{Notes}-----------------
        * This function is useful for listing the attributes of a module, which can be helpful for understanding the module's functionality.
    """
    attributes = inspect.getmembers(mod, lambda a: not (inspect.isroutine(a)))
    return [
        a
        for a in attributes
        if not (a[0].startswith("__") and a[0].endswith("__") or a[0].startswith("_"))
    ]


def get_attributes_pd(obj, filter_list=None):
    """
    * type-def ::(Any, Optional[List[str]]) -> pd.DataFrame
    * ---------------{Function}---------------
        * Get a DataFrame containing attributes for a given object, with optional filtering.
    * ----------------{Returns}---------------
        * : df ::pd.DataFrame | A DataFrame containing the attribute names and their values
    * ----------------{Params}----------------
        * : obj ::Any | The object to get attributes from
        * : filter_list ::Optional[List[str]] | A list of attribute names to exclude from the output DataFrame (default: None)
    * ----------------{Usage}-----------------
        * >>> import numpy as np
        * >>> filter_list = ['__name__', '__package__']
        * >>> attributes_df = get_attributes_pd(np, filter_list)
        * >>> print(attributes_df)
    * ----------------{Output}----------------
        *      0                 1
        * 0  ALLOW_THREADS       1
        * 1  BUFSIZE          8192
        * 2  CLIP               0
        * ...
    * ----------------{Notes}-----------------
        * This function is useful for generating a DataFrame containing the attributes of an object, which can be helpful for understanding the object's functionality.
    """
    if filter_list:
        filtered_attr = filter_tuples(get_attributes(obj), filter_list)
        df = pd.DataFrame(filtered_attr).T
    else:
        df = pd.DataFrame(get_attributes(obj)).T
    return pd_row_header(df)


def get_nested_attributes_pd(data, filter_list=None, axis=0):
    """
    * type-def ::(List[object], List[str], int) -> pd.DataFrame
    * ---------------{Function}---------------
        * Get attributes of a list of objects in a concatenated pandas DataFrame
    * ----------------{Returns}---------------
        * : df ::pd.DataFrame | A DataFrame containing attributes and their values
    * ----------------{Params}----------------
        * : data ::List[object] | A list of objects to extract attributes from
        * : filter_list ::List[str] | A list of attribute names to filter out (default: None)
        * : axis ::int | Axis along which to concatenate the DataFrames (default: 0)
    * ----------------{Usage}-----------------
        * >>> class MyClass:
        * ...     def __init__(self, x, y):
        * ...         self.x = x
        * ...         self.y = y
        * >>> obj_list = [MyClass(1, 2), MyClass(3, 4), MyClass(5, 6)]
        * >>> df = get_nested_attributes_pd(obj_list)
    * ----------------{Output}----------------
        *     0         1
        * 0  attr1    value1
        * 1  attr2    value2
        * 2  attr3    value3
        * ...
        * 0  attr1    value1
        * 1  attr2    value2
        * 2  attr3    value3
        * ...
    """
    for i, section in enumerate(data):
        if i == 0:
            sections = get_attributes_pd(section, filter_list)
        else:
            temp = get_attributes_pd(section, filter_list)
            sections = pd.concat((sections, temp), axis=axis)
    return sections


def concat_pd_list(data, axis=0):
    """
    * type-def ::(List[pd.DataFrame], int) -> pd.DataFrame
    * ---------------{Function}---------------
        * Concatenate a list of pandas DataFrames along a specified axis
    * ----------------{Returns}---------------
        * : df ::pd.DataFrame | A concatenated DataFrame
    * ----------------{Params}----------------
        * : data ::List[pd.DataFrame] | A list of DataFrames to concatenate
        * : axis ::int | Axis along which to concatenate the DataFrames (default: 0)
    * ----------------{Usage}-----------------
        * >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        * >>> df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})
        * >>> df3 = pd.DataFrame({'A': [9, 10], 'B': [11, 12]})
        * >>> df_list = [df1, df2, df3]
        * >>> concatenated_df = concat_pd_list(df_list)
    * ----------------{Output}----------------
        *     0         1
        * 0  attr1    value1
        * 1  attr2    value2
        * 2  attr3    value3
        * ...
        * 0  attr1    value1
        * 1  attr2    value2
        * 2  attr3    value3
        * ...
    """
    for i, section in enumerate(data):
        if i == 0:
            sections = section
        else:
            sections = pd.concat((sections, section), axis=axis)
    return sections


# {Row to column header}#


def pd_row_header(df, idx=0):
    """
    * type-def ::(pd.DataFrame, int) -> pd.DataFrame
    * ---------------{Function}---------------
        * Set the DataFrame column headers using the values of a specific row
    * ----------------{Returns}---------------
        * : df ::pd.DataFrame | A DataFrame with updated column headers
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The input DataFrame
        * : idx ::int | The index of the row to use for column headers (default: 0)
    * ----------------{Usage}-----------------
        * >>> data = {"A": ["header1", 1, 2], "B": ["header2", 3, 4]}
        * >>> df = pd.DataFrame(data)
        * >>> new_df = pd_row_header(df)
    * ----------------{Output}----------------
        *   header1  header2
        * 1       1        3
        * 2       2        4
    """
    df.columns = df.iloc[idx]
    return df.reindex(df.index.drop(idx))


def pd_split_lazy(df, column):
    """
    * type-def ::(pd.DataFrame, str) -> Iterator[Tuple[str, ...]]
    * ---------------{Function}---------------
        * Lazily split the values of a DataFrame column containing strings
    * ----------------{Returns}---------------
        * : iterator ::Iterator[Tuple[str, ...]] | An iterator yielding tuples of split values
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The input DataFrame
        * : column ::str | The name of the column to split
    * ----------------{Usage}-----------------
        * >>> data = {"A": ["one two", "three four"], "B": [1, 2]}
        * >>> df = pd.DataFrame(data)
        * >>> split_iterator = pd_split_lazy(df, "A")
    * ----------------{Output}----------------
        * ("one", "two")
        * ("three", "four")
    """
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
    """
    * type-def ::(np.ndarray) -> np.ndarray
    * ---------------{Function}---------------
        * Reshape a multi-dimensional numpy array to a 1-row 2D array
    * ----------------{Returns}---------------
        * : reshaped_data ::np.ndarray | A reshaped numpy array
    * ----------------{Params}----------------
        * : data ::np.ndarray | The input numpy array to reshape
    * ----------------{Usage}-----------------
        * >>> data = np.array([[1, 2], [3, 4]])
        * >>> reshaped_data = reshape_multi_dimensional(data)
    * ----------------{Output}----------------
        * array([[1, 2, 3, 4]])
    """
    return data.reshape(1, data.shape[0])


def ndistinct(x):
    """
    * type-def ::(np.ndarray) -> int
    * ---------------{Function}---------------
        * Count the number of distinct values in a numpy array
    * ----------------{Returns}---------------
        * : out ::int | The number of distinct values in the input array
    * ----------------{Params}----------------
        * : x ::np.ndarray | The input numpy array
    * ----------------{Usage}-----------------
        * >>> x = np.array([1, 2, 3, 1, 2, 3, 4])
        * >>> distinct_count = ndistinct(x)
    * ----------------{Output}----------------
        * 4
    """
    out = len(np.unique(x))
    return out


def calculate_bounds_and_iterations(size, increments):
    """
    * type-def ::(int, int) -> Tuple[int, int, int]
    * ---------------{Function}---------------
        * Calculate lower and upper bounds and the number of iterations for dividing a size into increments
    * ----------------{Returns}---------------
        * : lb ::int | The lower bound
        * : ub ::int | The upper bound
        * : its ::int | The number of iterations
    * ----------------{Params}----------------
        * : size ::int | The total size to be divided
        * : increments ::int | The size of increments
    * ----------------{Usage}-----------------
        * >>> size = 100
        * >>> increments = 10
        * >>> lb, ub, its = calculate_bounds_and_iterations(size, increments)
    * ----------------{Output}----------------
        * (0, 10, 10)
    """
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
    """
    * type-def ::(pd.DataFrame, int) -> List[pd.DataFrame]
    * ---------------{Function}---------------
        * Split a DataFrame into a specified number of sections
    * ----------------{Returns}---------------
        * : result ::List[pd.DataFrame] | A list of DataFrames, each representing a section of the original DataFrame
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The input DataFrame to be split
        * : sections ::int | The number of sections to split the DataFrame into (default: 10)
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({'A': range(100), 'B': range(100)})
        * >>> split_dfs = split_dataframe(df, sections=5)
    * ----------------{Output}----------------
        * [    A   B
              ..
            ,    A   B
              ..
            , ... ]
    """
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


def chunk_dataframe(df, sections=10):
    """
    * type-def ::(pd.DataFrame, int) -> Generator[pd.DataFrame]
    * ---------------{Function}---------------
        * Yield chunks of a DataFrame given a specified number of sections
    * ----------------{Returns}---------------
        * : temp_df ::pd.DataFrame | A chunk of the original DataFrame
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The input DataFrame to be chunked
        * : sections ::int | The number of sections to divide the DataFrame into (default: 10)
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({'A': range(100), 'B': range(100)})
        * >>> for chunk in chunk_dataframe(df, sections=5):
        * ...     print(chunk)
    * ----------------{Output}----------------
        *     A   B
          ...
        *     A   B
          ...
        * ...
    """
    part_size = get_stride_len(df.shape[0], sections)
    lb = 0
    ub = part_size
    for i in range(sections):
        temp_df = df[lb:ub].copy()
        lb = lb + part_size
        ub = ub + part_size
        yield temp_df


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
    """
    * type-def ::(Connection, str, str, str) -> str
    * ---------------{Function}---------------
        * Construct a SQL query to perform an inner join between two tables based on a common column
    * ----------------{Returns}---------------
        * : result ::str | A SQL query string for inner join
    * ----------------{Params}----------------
        * : con ::Connection | A connection object to the database
        * : table1 ::str | The name of the first table to join
        * : table2 ::str | The name of the second table to join
        * : col ::str | The name of the common column to join on
    * ----------------{Usage}-----------------
        * >>> query = inner_join(con, "table1", "table2", "id")
        * >>> print(query)
    * ----------------{Output}----------------
        * SELECT table1.column1, table1.column2, table2.column1 as table2_column1, table2.column2 as table2_column2
          FROM table1 INNER JOIN table2 ON table1.id=table2.id
    """
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
    """
    * type-def ::(Connection, str, str, str, str, Optional[str]) -> str
    * ---------------{Function}---------------
        * Construct a SQL query to perform an inner join between a subquery (or a view) and a table based on a common column
    * ----------------{Returns}---------------
        * : result ::str | A SQL query string for inner join with subquery
    * ----------------{Params}----------------
        * : con ::Connection | A connection object to the database
        * : query ::str | The SQL query string for the subquery (or the view)
        * : view1 ::str | The alias for the subquery (or the view) to be used in the join
        * : table2 ::str | The name of the second table to join
        * : col ::str | The name of the common column to join on in the subquery (or the view)
        * : col2 ::Optional[str] | The name of the common column to join on in the second table (default: None, uses 'col' for both)
    * ----------------{Usage}-----------------
        * >>> subquery = "SELECT * FROM table1 WHERE value > 100"
        * >>> query = nest_inner(con, subquery, "view1", "table2", "id")
        * >>> print(query)
    * ----------------{Output}----------------
        * SELECT view1.column1, view1.column2, table2.column1 as table2_column1, table2.column2 as table2_column2
          FROM (SELECT * FROM table1 WHERE value > 100) as view1 INNER JOIN table2 ON view1.id=table2.id
    """

    try:
        pd.read_sql(query + " limit 1", con)
        pd.read_sql("select * from {} limit 1;".format(table2), con)
    except Exception as e:
        raise ValueError(f"Invalid SQL query or table name: {e}")

    if not (isinstance(col, str) and (col2 is None or isinstance(col2, str))):
        raise ValueError("Invalid column name(s)")

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
    """
    * type-def ::(pd.DataFrame) -> Iterator[pd.Series]
    * ---------------{Function}---------------
        * Yields each row of the input DataFrame as a pandas Series
    * ----------------{Returns}---------------
        * : yield ::pd.Series | A single row of the DataFrame as a pandas Series
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The DataFrame to iterate over
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        * >>> for row in iter_range_pd(df):
        * ...     print(row)
    * ----------------{Output}----------------
        * A    1
          B    3
          Name: 0, dtype: int64
        * A    2
          B    4
          Name: 1, dtype: int64
    """
    for i in range(df.shape[0]):
        yield (df.iloc[i])


def pd_csv_to_buffer(data):
    """
    * type-def ::(pd.DataFrame) -> io.BytesIO
    * ---------------{Function}---------------
        * Converts a DataFrame to a CSV and writes it to a BytesIO buffer
    * ----------------{Returns}---------------
        * : out_buffer ::io.BytesIO | A BytesIO buffer containing the CSV data
    * ----------------{Params}----------------
        * : data ::pd.DataFrame | The DataFrame to be converted to CSV
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        * >>> buffer = pd_csv_to_buffer(df)
    * ----------------{Output}----------------
        * <_io.BytesIO object at 0x7f03a2e23dc0>
    """
    import io

    u = data.to_csv().encode("latin-1")
    out_buffer = io.BytesIO()
    out_buffer.seek(0)
    out_buffer.write(u)
    out_buffer.seek(0)
    return out_buffer


def max_len_aos(data):
    """
    * type-def ::(List[Dict[str, Any]]) -> Tuple[int, int]
    * ---------------{Function}---------------
        * Find the dictionary with the maximum number of keys in a list of dictionaries
    * ----------------{Returns}---------------
        * : max_len ::Tuple[int, int] | A tuple containing the index and the maximum number of keys
    * ----------------{Params}----------------
        * : data ::List[Dict[str, Any]] | A list of dictionaries to be checked
    * ----------------{Usage}-----------------
        * >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}]
        * >>> max_len = max_len_aos(data)
    * ----------------{Output}----------------
        * (1, 3)
    * ----------------{Notes}-----------------
        * This function can be useful when working with a list of dictionaries with varying numbers of keys, especially when you want to identify the most complete record or perform other operations based on the maximum number of keys.
    """
    max_len = (0, 0)
    for idx, i in enumerate(data):
        current_len = len(i.keys())
        if current_len > max_len[1]:
            max_len = (idx, current_len)
    return max_len


def transform_aos_soa(dict_list):
    """
    * type-def ::(List[Dict[str, Any]]) -> Dict[str, List[Any]]
    * ---------------{Function}---------------
        * Transform a list of dictionaries (array of structs) into a dictionary of lists (struct of arrays)
    * ----------------{Returns}---------------
        * : soa ::Dict[str, List[Any]] | A dictionary with keys as column names and values as lists of data
    * ----------------{Params}----------------
        * : dict_list ::List[Dict[str, Any]] | A list of dictionaries to be transformed
    * ----------------{Usage}-----------------
        * >>> data = [{'a': 1, 'b': 2}, {'a': 3, 'b': 4, 'c': 5}]
        * >>> soa = transform_aos_soa(data)
    * ----------------{Output}----------------
        * {'a': [1, 3], 'b': [2, 4], 'c': [None, 5]}
    * ----------------{Notes}-----------------
        * This function is useful when working with data that needs to be transformed from an array of structs to a struct of arrays format. This is often done to improve the performance of data processing tasks and to ensure compatibility with certain libraries and APIs that expect data in a specific format.
    """
    # max_len = max_len_aos(dict_list)
    # cols = list(dict_list[max_len[0]].keys())
    # soa = {}
    # for i in cols:
    #     soa[i] = []
    #     for j in range(len(dict_list)):

    #         try:
    #             soa[i].append(dict_list[j][i])
    #         except Exception:
    #             soa[i].append(None)
    # return soa
    soa = {}
    for d in dict_list:
        for k, v in d.items():
            if k not in soa:
                soa[k] = [None] * len(dict_list)
            soa[k].append(v)

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
    """
    * type-def ::(Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]
    * ---------------{Function}---------------
        * Get rows or elements with NaN values in a pandas DataFrame or Series
    * ----------------{Returns}---------------
        * : nan_rows ::Union[pd.Series, pd.DataFrame] | A DataFrame or Series with rows or elements containing NaN values
    * ----------------{Params}----------------
        * : df ::Union[pd.Series, pd.DataFrame] | The input DataFrame or Series to filter
    * ----------------{Usage}-----------------
        * >>> data = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
        * >>> nan_rows = pd_get_nan(data)
    * ----------------{Output}----------------
        *      A    B
        * 2  NaN  6.0
        * 1  2.0  NaN
    * ----------------{Notes}-----------------
        * This function is useful for filtering out missing data points in a DataFrame or Series. It can help identify rows or elements that need further attention or preprocessing.
    """
    if type(df) == pd.core.series.Series:
        return df[df.isna()]
    else:
        return df[df.isna()[df.columns[0]]]


def pd_get_not_nan(df):
    """
    * type-def ::(Union[pd.Series, pd.DataFrame]) -> Union[pd.Series, pd.DataFrame]
    * ---------------{Function}---------------
        * Get rows or elements without NaN values in a pandas DataFrame or Series
    * ----------------{Returns}---------------
        * : not_nan_rows ::Union[pd.Series, pd.DataFrame] | A DataFrame or Series with rows or elements without NaN values
    * ----------------{Params}----------------
        * : df ::Union[pd.Series, pd.DataFrame] | The input DataFrame or Series to filter
    * ----------------{Usage}-----------------
        * >>> data = pd.DataFrame({'A': [1, 2, np.nan], 'B': [4, np.nan, 6]})
        * >>> not_nan_rows = pd_get_not_nan(data)
    * ----------------{Output}----------------
        *      A    B
        * 0  1.0  4.0
    * ----------------{Notes}-----------------
        * This function is useful for filtering out rows or elements without missing data points in a DataFrame or Series. It can help clean up the data for further processing or analysis.
    """
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


def pd_get_dummies_concat(source_df, column, prefix=None, drop=True):
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
    result = pd.concat(
        [source_df, pd.get_dummies(source_df[column], prefix=prefix)],
        axis=1,
        sort=False,
    )
    if drop:
        result.drop(column, axis=1, inplace=True)
    return result


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
        df[col][df[col].str.contains(contained_string)].index,
        inplace=True,
    )
    return df


def partition_df(df, col):
    for current_element in df[col].unique():
        result = df[df[col] == current_element]
        yield result


def contains_extract_pd(df, contained_string):
    extracted_rows = df[df.ClientRequestPath.str.contains(contained_string)]
    df = df.drop(kendo_rows.index, axis=0)
    return df, extracted_rows


def np_startswith_indeces(array, data):
    return np.flatnonzero(np.char.startswith(array, data))


def np_startswith_mask(array, data):
    return np.char.startswith(array, data)


def np_startswith_values(array, data):
    return array[(np.char.startswith(array, data))]


def np_drop_nan(data):
    return data[~np.isnan(data)]


def pd_train_test_val_split(df, col):
    y = df[col]
    X = pd_except_cols(df, [col])
    X_train, X_test, y_train, y_test = sk.model_selection.train_test_split(
        X, y, test_size=0.20
    )
    X_train, X_val, y_train, y_val = sk.model_selection.train_test_split(
        X_train, y_train, test_size=0.25
    )  # 0.25 * 0.8 = 0.2
    return X_train, X_test, X_val, y_train, y_test, y_val


def pd_except_cols(df, col_list):
    return df[df.columns.difference(col_list)]


def pd_distance_matrix(df):
    from scipy.spatial import distance_matrix

    df = df.select_dtypes("number")
    return pd.DataFrame(
        distance_matrix(df.values, df.values), index=df.index, columns=df.index
    )


def np_topk(data, k):
    return data.argsort()[-k:][::-1]


def x_only_train_test_split(data, test_size=0.33):

    from sklearn.model_selection import train_test_split

    X_train, X_test, _, _ = train_test_split(
        data, np.zeros(data.shape[0]), test_size=test_size, random_state=42
    )
    return X_train, X_test


def pd_scale_norm_df(df):
    import sklearn as sk

    dtype_cols = [i for i in df.columns if df[i].dtype == np.object]
    cols_to_norm = df.loc[:, ~df.columns.isin(dtype_cols)].columns
    train = df.copy()
    train[cols_to_norm] = sk.preprocessing.StandardScaler().fit_transform(
        train[cols_to_norm]
    )
    return df


def zeroper(df, value=0):
    """
    helper function to print out percentage of zeroes by column
    Useful for optimizations based on usage of a sparse matrix
    """
    l = []
    columns = []
    for i in range(len(df.columns)):
        if 0 in df[df.columns[i]].value_counts():
            if (
                100 * df[df.columns[i]].value_counts().loc[0] / len(df[df.columns[i]])
                > value
            ):
                l.append(
                    (
                        df.columns[i],
                        100
                        * df[df.columns[i]].value_counts().loc[0]
                        / len(df[df.columns[i]]),
                    )
                )
            else:
                pass
        else:
            pass
    print("-" * 55)
    for j in range(len(l)):
        columns.append(l[j][0])
        print("Percent of zeroes: ", l[j])
        print("-" * 55)
    return (columns, l)


def pd_get_noncategorical(df):
    return df.loc[:, ~df.columns.isin(df.select_dtypes("number").columns)]


def pd_h_concat(df_1, df_2, cols=None, keep_col=False):
    if cols is not None:
        df_2 = df_2[cols]
    df = pd.concat((df_1, df_2), axis=1, verify_integrity=True)
    if cols is not None and keep_col == False:
        df.drop(cols, axis=1, inplace=True)
    return df


def pct_col_nul(df, col):
    print(
        f"""Dataframe col {col} Nulls accounts for
          {(df.shape[0] - df[col].dropna().shape[0]) * 100 / df.shape[0]:.2f}%
          of the whole DataFrame of len={df.shape[0]}"""
    )


def np_deconstructed_mask(data: np.array, key, matching_key):
    """
    * type-def ::(np.array, Any, Any) -> np.array
    * ---------------{Function}---------------
        * Create a boolean mask for an array of dictionaries based on key-value matching
    * ----------------{Returns}---------------
        * : mask ::np.array | A boolean mask where each element is True if the dictionary has the given key and its value matches the matching_key
    * ----------------{Params}----------------
        * : data ::np.array | The input numpy array containing dictionaries
        * : key ::Any | The key to search for in the dictionaries
        * : matching_key ::Any | The value that the key should match
    * ----------------{Usage}-----------------
        * >>> data_np = np.array([{'eventType': 'A', 'value': 1}, {'eventType': 'B', 'value': 2}, {'eventType': 'A', 'value': 3}])
        * >>> mask = np_deconstructed_mask(data_np, 'eventType', 'A')
    * ----------------{Output}----------------
        * array([ True, False,  True])
    * ----------------{Notes}-----------------
        * This function is useful when working with numpy arrays containing dictionaries, and you need to create a boolean mask based on a key-value match.
        * np.vectorize(lambda x: x["eventType"] == uniq[0])(data_np)
    """
    if isinstance(data, list):
        data = np.array(data)

    return np.vectorize(lambda x: x[key] == matching_key)(data)


def pd_list_to_pd(result, batches):
    from math import ceil

    chunk_len = int(ceil(len(result)) / batches)
    final_df_result = []
    for batch in range(batches):
        temp_batch = batch * chunk_len
        for i in tqdm(range(chunk_len)):
            if i == 0:
                final_df = result[temp_batch]
            else:
                final_df = pd.concat([final_df, result[temp_batch + i]], axis=0)
        final_df_result.append(final_df)
    for idx, temp_df in tqdm(enumerate(final_df_result)):
        if idx == 0:
            final_df = temp_df
        else:
            final_df = pd.concat([final_df, temp_df], axis=0)
    final_df.reset_index(inplace=True, drop=True)
    return final_df


def group_sort_unique(df, group_by, sort_by, unique=True):
    if unique:
        return (
            df.groupby(group_by)
            .apply(lambda x: len(x[sort_by].unique()))
            .sort_values(ascending=False)
        )
    else:
        return (
            df.groupby(group_by)
            .apply(lambda x: len(x[sort_by]))
            .sort_values(ascending=False)
        )


def pd_table_fmt(df):
    df = df.fillna("-")
    cols = list(df.columns)
    contents = df.to_numpy()
    return [cols, [list(i) for i in contents]]


def dt_features(df, label=None):
    """
    Creates time series features from datetime index.
    """
    df = df.copy()
    df["date"] = df.index
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.weekofyear

    X = df[
        [
            "hour",
            "dayofweek",
            "quarter",
            "month",
            "year",
            "dayofyear",
            "dayofmonth",
            "weekofyear",
        ]
    ]
    if label:
        y = df[label]
        return X, y
    return X


def datetime_features_inplace(df):
    """
    Creates time series features from datetime index.
    """
    df["date"] = df.index
    df["hour"] = df["date"].dt.hour
    df["dayofweek"] = df["date"].dt.dayofweek
    df["quarter"] = df["date"].dt.quarter
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["dayofyear"] = df["date"].dt.dayofyear
    df["dayofmonth"] = df["date"].dt.day
    df["weekofyear"] = df["date"].dt.weekofyear

    return df


def compare_dataframes(df1, df2, column_name1, column_name2):
    """
    Compare two DataFrames and return the rows that are missing in one DataFrame.

    Args:
        df1 (pd.DataFrame): The first DataFrame.
        df2 (pd.DataFrame): The second DataFrame.
        column_name1 (str): The name of the column in df1 to use for comparison.
        column_name2 (str): The name of the column in df2 to use for comparison.

    Returns:
        pd.DataFrame: A DataFrame containing the rows that are missing in one of the input DataFrames.
    """

    # Merge the two DataFrames and keep all rows
    merged_df = df1.merge(df2, left_on=column_name1, right_on=column_name2, how="outer")

    # Get the rows from df1 that are not in df2
    missing_rows = merged_df[merged_df[column_name2].isnull()]

    # Print the number of rows in each DataFrame
    print(f"Number of rows in df1: {len(df1)}")
    print(f"Number of rows in df2: {len(df2)}")
    print(f"Number of missing rows: {len(missing_rows)}")

    # Return the missing rows
    return missing_rows


def LOOEncoding(
    df,
    cols,
    target,
    sigma=0.05,
    inference=False,
    bucket_name="",
    blob_name="",
    project_name="",
):
    """
    * type-def ::(pd.DataFrame, List[str], str, float, bool, str, str, str) -> pd.DataFrame
    * ---------------{Function}---------------
        * Performs leave-one-out (LOO) encoding on the given dataframe
    * ----------------{Returns}---------------
        * : df_encoded ::pd.DataFrame | A pandas dataframe with the encoded columns
    * ----------------{Params}----------------
        * : df ::pd.DataFrame | The pandas dataframe to encode
        * : cols ::List[str] | A list of columns to encode
        * : target ::str | The target column to use for encoding
        * : sigma ::float | The standard deviation to use for LOO encoding (default: 0.05)
        * : inference ::bool | A boolean indicating whether the function is being called for inference or training (default: False)
        * : bucket_name ::str | The name of the cloud storage bucket to use for caching the fitted encoder (default: "")
        * : blob_name ::str | The name of the cloud storage blob to use for caching the fitted encoder (default: "")
        * : project_name ::str | The name of the Google Cloud project to use for caching the fitted encoder (default: "")
    * ----------------{Usage}-----------------
        * >>> df = pd.DataFrame({'A': [1, 2, 1], 'B': [1, 1, 2], 'target': [10, 20, 30]})
        * >>> encoded_df = LOOEncoding(df, cols=['A', 'B'], target='target')
    * ----------------{Output}----------------
        * A DataFrame with encoded 'A' and 'B' columns
    * ----------------{Notes}-----------------
        * This function is particularly useful for encoding categorical features with high cardinality.
        * The function uses the LeaveOneOutEncoder from the Category Encoders library.
        * The fitted encoder is cached in a cloud storage bucket for future use during inference.
    """
    import dill as pickle
    from io import BytesIO
    import joblib
    import cloudpickle

    y = df[target]
    X = pd.DataFrame(df, columns=cols)

    if not inference:
        # Fit the LeaveOneOutEncoder using the given columns and target
        loo_encoder = ce.LeaveOneOutEncoder(df, cols=cols, sigma=sigma)
        loo_encoder.fit(X, y)

        # Save the fitted encoder to a cloud storage bucket
        encoder_bytes = cloudpickle.dumps(loo_encoder)

        bytes_container = BytesIO()
        joblib.dump(loo_encoder, bytes_container)
        bytes_container.seek(0)  # update to enable reading

        upload_bytesio_blob(
            bucket_name, blob_name, encoder_bytes, project_name, encode=False
        )
        print(
            f"Saved fitted encoder to bucket '{bucket_name}', blob '{blob_name}', project '{project_name}'"
        )
        upload_bytesio_blob(
            bucket_name,
            blob_name + "_joblib",
            bytes_container.read(),
            project_name,
            encode=False,
        )
        print(
            f"Saved fitted encoder to bucket '{bucket_name}', blob '{blob_name+ '_joblib'}', project '{project_name}'"
        )
    else:

        try:
            # Download the encoder from the cloud storage bucket
            encoder_bytes = download_bytesio_blob(bucket_name, blob_name, project_name)

            loo_encoder = pickle.loads(encoder_bytes.getbuffer())
        except Exception as e:
            exc_type, exc_obj, exc_tb = sys.exc_info()
            fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
            print(exc_type, fname, exc_tb.tb_lineno)

            encoder_bytes_jl = download_bytesio_blob(
                bucket_name, blob_name + "_joblib", project_name
            )

            loo_encoder = joblib.load(encoder_bytes_jl)

        print(
            f"Downloaded encoder from bucket '{bucket_name}', blob '{blob_name}', project '{project_name}'"
        )

    # Transform the data using the fitted encoder
    X_encoded = loo_encoder.transform(X)
    return X_encoded


def uneven_binning(df, col, first_half=100, second_half=10):
    """
    Splits a numerical column of a DataFrame into uneven bins.

    This function takes in a DataFrame, the name of a numerical column in that DataFrame, and the number of bins
    to use for the first and second halves of the column's values. It generates evenly split ranges for the first
    half of the values and for the second half of the values using the `np.linspace` function, and then combines
    these ranges into a single list of bins. It then uses the `pd.cut` function to split the column's values into
    the bins and returns a new categorical series with the split values.

    Args:
        df: The DataFrame containing the column to be split into bins.
        col: The name of the numerical column to be split into bins.
        first_half: The number of bins to use for the first half of the column's values.
        second_half: The number of bins to use for the second half of the column's values.

    Returns:
        A categorical series with the values from the original column split into the specified bins.
    """
    # Use the np.linspace() function to generate 100 evenly split ranges for the first half of the values
    bins_first_half = np.linspace(df[col].min(), df[col].max() / 2, 100)

    # Use the np.linspace() function to generate 10 evenly split ranges for the second half of the values
    bins_second_half = np.linspace(df[col].max() / 2, df[col].max(), 10)

    # Combine the bins for the first and second halves into a single list of bins
    bins = [-1] + list(bins_first_half) + list(bins_second_half)
    bins = sorted(bins)

    # Check the values of the bins to make sure they are sorted in the correct order
    if not all(bins[i] <= bins[i + 1] for i in range(len(bins) - 1)):
        raise ValueError("Bins are not sorted in the correct order")

    u = pd.cut(df[col], bins=bins, labels=None, duplicates="drop")

    return u.cat.rename_categories([col + str(x) for x in u.cat.categories])


def create_cyclical_features(df, datetime_col, inplace=True):
    """
    Creates cyclical features from a datetime column in a dataframe.

    This function takes in a dataframe, the name of a datetime column in the dataframe, and a boolean indicating
    whether the original columns should be replaced or not. It extracts the month, day, and weekday from the
    datetime column and creates sin and cosine-based features for these columns, as well as radial basis functions
    for these columns. It then drops the original month, day, and weekday columns and returns the modified dataframe.

    Args:
        df: A Pandas dataframe containing a datetime column.
        datetime_col: The name of the datetime column in the dataframe.
        inplace: A boolean indicating whether the original columns should be replaced or not.

    Returns:
        The modified dataframe with the added cyclical features.
    """
    # Convert the datetime column to datetime type
    if df[datetime_col].dtype == "datetime64[ns]":
        df[datetime_col] = pd.to_datetime(df[datetime_col])
    else:
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")

    # Extract the month, day, and weekday from the datetime column
    df["month"] = df[datetime_col].dt.month
    df["day"] = df[datetime_col].dt.day
    df["weekday"] = df[datetime_col].dt.weekday

    try:
        # Create sine and cosine-based basis functions for the month, day, and weekday using numpy
        df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
        df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
        df["sin_day"] = np.sin(2 * np.pi * df["day"] / 31)
        df["cos_day"] = np.cos(2 * np.pi * df["day"] / 31)
        df["sin_weekday"] = np.sin(2 * np.pi * df["weekday"] / 7)
        df["cos_weekday"] = np.cos(2 * np.pi * df["weekday"] / 7)
        # Create radial basis functions for the month, day, and weekday
        df["rbf_month"] = np.exp(-1 * (df["month"] / 6) ** 2)
        df["rbf_day"] = np.exp(-1 * (df["day"] / 15.5) ** 2)
        df["rbf_weekday"] = np.exp(-1 * (df["weekday"] / 3.5) ** 2)

    except:
        # Create sine and cosine-based basis functions for the month, day, and weekday using Pandas
        df["sin_month"] = df["month"].apply(lambda x: np.sin(2 * np.pi * x / 12))
        df["cos_month"] = df["month"].apply(lambda x: np.cos(2 * np.pi * x / 12))
        df["sin_day"] = df["day"].apply(lambda x: np.sin(2 * np.pi * x / 31))
        df["cos_day"] = df["day"].apply(lambda x: np.cos(2 * np.pi * x / 31))
        df["sin_weekday"] = df["weekday"].apply(lambda x: np.sin(2 * np.pi * x / 7))
        df["cos_weekday"] = df["weekday"].apply(lambda x: np.cos(2 * np.pi * x / 7))
        # Create radial basis functions for the month, day, and weekday
        df["rbf_month"] = df["month"].apply(
            lambda x: np.exp(-1 * (df["month"] / 6) ** 2)
        )
        df["rbf_day"] = df["day"].apply(lambda x: np.exp(-1 * (df["day"] / 15.5) ** 2))
        df["rbf_weekday"] = df["weekday"].apply(
            lambda x: np.exp(-1 * (df["weekday"] / 3.5) ** 2)
        )

    # Drop the original month, day, and weekday columns
    if inplace:
        df.drop(["month", "day", "weekday"], axis=1, inplace=True, errors="ignore")
    else:
        df = df.drop(["month", "day", "weekday"], axis=1, errors="ignore")
    return df


def check_for_nans(df):
    rows_to_drop = {}
    for col in df.columns:
        if df[col].isna().any():
            print(f"Column {col} has NaNs or None values in the following row:")
            row_with_nans = df[df[col].isna()]
            print(row_with_nans.sample(1))
            rows_to_drop[col] = list(row_with_nans.index)
            print()

    return rows_to_drop


def fill_nans(df, rows_to_drop, method="mean", drop_method="drop"):
    for col, rows in rows_to_drop.items():
        print(f"Column {col} has NaNs or None values in the following rows:")
        print(df.loc[rows])
        if drop_method == "drop":
            df.drop(rows, inplace=True)
            print(
                f"The rows with NaNs or None values in column {col} have been dropped."
            )
        elif drop_method == "fill":
            print(
                f"NaNs or None values in column {col} have been filled using the '{method}' method."
            )
            if method == "mean" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].mean(), inplace=True)
            elif method == "median" and pd.api.types.is_numeric_dtype(df[col]):
                df[col].fillna(df[col].median(), inplace=True)
            elif method == "mode" and pd.api.types.is_string_dtype(df[col]):
                df[col].fillna(df[col].mode().iloc[0], inplace=True)
            elif method == "zero":
                df[col].fillna(0, inplace=True)
            else:
                df[col].fillna(np.nan, inplace=True)
        else:
            print(
                f"Invalid drop method: {drop_method}. The rows with NaNs or None values in column {col} have not been dropped."
            )
    return df


def load_csv_from_url(url: str) -> pd.DataFrame:
    """
    * ---------------Function---------------
    * Load a CSV file from a given URL and return a pandas DataFrame object
    * ----------------Returns---------------
    * -> pd.DataFrame: A pandas DataFrame object containing the data from the CSV file
    * ----------------Params----------------
    * url: str: The URL of the CSV file to be loaded
    * ----------------Usage-----------------
    * load_csv_from_url('https://example.com/data.csv')
    * ----------------Notes-----------------
    * This function sends a GET request to the given URL with a specific User-Agent header to mimic an iPhone browser. If the request is successful (200 status code), it returns a pandas DataFrame object containing the data from the CSV file. Otherwise, it raises an Exception.
    """
    import requests

    headers = {
        "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 13_2_3 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/13.0.3 Mobile/15E148 Safari/604.1"
    }
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return pd.read_csv(url)
    else:
        raise Exception("Failed to load the CSV file from the given URL.")


def align_data(data):
    """
    Return a copy of the array that is aligned in memory.
    Performance: Improves cache utilization and memory access speed.
    Stability: Reduces unexpected performance hits due to misalignment.
    """
    return np.ascontiguousarray(data)


def efficient_sum(x):
    """
    Efficiently sum elements of x using np.einsum.
    prob = efficient_sum(np.log1p((data[t:s] - muT) ** 2 * inv_nuT_scale))
    """
    return np.einsum("i->", x)


def load_memmap(filename, dtype="float32", mode="r", shape=None):
    """
    Load a memory-mapped array from a file.
    """
    return np.memmap(filename, dtype=dtype, mode=mode, shape=shape)


def efficient_hypot(a, b):
    """
    * ---------------Typedef----------------
    * type-def ::(a: <any>, b: <any>) -> float*
    * ---------------Function---------------
    * Computes the Euclidean norm efficiently.
    * This function calculates the square root of the sum of the squares of a and b.
    * It is more accurate for large or small values and has an optimized C implementation in NumPy.
    * ---------------Returns---------------
    * -> result ::float | The Euclidean norm sqrt(a**2 + b**2).
    * ---------------Params----------------
    * a ::<any> | The first input value.
    * b ::<any> | The second input value.
    * ---------------Usage-----------------
    * This function can be used to calculate the distance between two points in a 2D space.
    * For example: efficient_hypot(3, 4) would return 5.0.
    * ---------------Notes-----------------
    * This function uses the hypot function from NumPy, which is implemented in C for better performance.
    * It is more accurate than calculating the Euclidean norm manually, especially for large or small values.
    """
    return np.hypot(a, b)


# GOLDMINE: https://personal.math.ubc.ca/~cbm/aands/frameindex.htm


def Q_rsqrt_numpy(number):
    """Fast inverse square root using NumPy. DOOM sourced"""
    threehalfs = 1.5

    x2 = number * 0.5
    y = number

    # Evil floating point bit level hacking
    y_i = y.view(np.int32)  # Interpret the bits of y as int32
    y_i = np.int32(0x5F3759DF) - (y_i >> 1)
    y = y_i.view(np.float32)  # Interpret the bits back to float32

    # Newton-Raphson iteration
    y = y * (threehalfs - (x2 * y * y))

    return y


def fast_sqrt_numpy(number):
    """Fast square root approximation using NumPy."""
    y = number

    # Bit level hacking
    y_i = y.view(np.int32)
    y_i = np.int32(1 << 29) + (y_i >> 1) - np.int32(1 << 22)
    y = y_i.view(np.float32)

    # One iteration of Newton-Raphson method
    y = y - (y * y - number) * 0.5 / y

    return y


def fast_exp_numpy(x):
    """Fast exponential approximation using NumPy."""
    x = np.float32(x)
    exp_i = np.int32(x * 12102203 + 1065353216)
    exp_f = exp_i.view(np.float32)
    return exp_f


def fast_log_numpy(x):
    """Fast natural logarithm approximation using NumPy."""
    y = np.float32(x)
    y_i = y.view(np.int32)
    ln = (y_i - 1064866805) * 8.262958288192749e-8
    return ln


def fast_sin_numpy(x):
    """Fast sine approximation using NumPy."""
    x = np.float32(x % (2 * np.pi))  # Wrap x within [0, 2π)

    # Use np.where to handle array inputs
    x = np.where(x > np.pi, x - 2 * np.pi, x)

    B = 1.27323954  # 4 / π
    C = -0.40528473  # -4 / π²
    y = B * x + C * x * np.abs(x)

    # Approximation improvement
    P = 0.225
    y = P * (y * np.abs(y) - y) + y
    return y


def fast_sin_lut(x, num_points=1024):
    """Fast sine approximation using a lookup table."""
    # Normalize x to the range [0, 2π]
    x = x % (2 * np.pi)

    # Generate a lookup table
    table = np.sin(np.linspace(0, 2 * np.pi, num_points, endpoint=False))

    # Map x to indices in the table
    indices = (x / (2 * np.pi) * num_points).astype(int) % num_points

    return table[indices]


def fast_sin_numpy_fifth(x):
    """Fast sine approximation using a 5th-degree polynomial."""
    # Normalize x to the range [-π, π]
    x = ((x + np.pi) % (2 * np.pi)) - np.pi

    # Use polynomial approximation
    x2 = x * x
    y = x * (0.9999966 - x2 * (0.166648 - x2 * (0.0083063 - x2 * 0.00018363)))

    return y


def fast_cos_numpy(x):
    """Fast cosine approximation using NumPy."""
    return fast_sin_numpy(x + np.pi / 2)


def fast_tan_numpy(x):
    """Fast tangent approximation using NumPy."""
    sin_x = fast_sin_numpy(x)
    cos_x = fast_cos_numpy(x)
    return sin_x / cos_x


def fast_exp2_numpy(x):
    """Fast base-2 exponential approximation using NumPy."""
    x = np.float32(x)
    exp_i = np.int32(x * 8388608 + 1065353216)
    exp_f = exp_i.view(np.float32)
    return exp_f


def fast_log2_numpy(x):
    """Fast base-2 logarithm approximation using NumPy."""
    y = np.float32(x)
    y_i = y.view(np.int32)
    exponent = ((y_i >> 23) & 255) - 127
    mantissa = (y_i & 0x7FFFFF) | 0x800000
    mantissa = mantissa / float(1 << 23)
    return exponent + mantissa - 1.0


def fast_pow(x, y):
    """Fast power approximation using NumPy."""
    # Approximate x^y uses the identity x^y = e^(y * ln(x))

    x = np.float32(x)
    y = np.float32(y)
    log_x = fast_log_numpy(x)
    return fast_exp_numpy(y * log_x)


def fast_sigmoid_numpy(x):
    """Fast sigmoid approximation using NumPy."""
    return 0.5 * x / (1 + np.abs(x)) + 0.5


def fast_inverse_numpy(x):
    """Fast inverse approximation using NumPy.
    TODO: Still slower
    """
    x = np.float32(x)
    y_i = x.view(np.int32)
    y_i = 0x7EEEEBB3 - y_i
    y = y_i.view(np.float32)
    return y


def fast_isqrt_numpy(x):
    """Fast integer square root approximation using NumPy.
    TODO: not quite there yet"""
    x = np.asarray(x, dtype=np.uint32)
    result = np.zeros_like(x, dtype=np.uint32)
    bit = np.uint32(1 << 30)

    bit = np.where(bit > x, bit >> 2, bit)

    for _ in range(16):  # Maximum 16 iterations for 32-bit integers
        temp = result + bit
        cond = x >= temp
        x = np.where(cond, x - temp, x)
        result = np.where(cond, (result >> 1) + bit, result >> 1)
        bit >>= 2
    return result


def fast_cbrt_numpy(x):
    """Fast cube root approximation using NumPy."""
    x = np.float32(x)
    x_third = x / 3.0
    y_i = x.view(np.int32)
    y_i = y_i // 3 + 709921077
    y = y_i.view(np.float32)
    return y


def fast_log2_numpy(x):
    """Fast base-2 logarithm approximation using NumPy."""
    x = np.float32(x)
    y_i = x.view(np.int32)
    exponent = ((y_i >> 23) & 255) - 127
    mantissa = (y_i & 0x7FFFFF) | 0x800000
    mantissa = mantissa / float(1 << 23)
    return exponent + mantissa - 1.0


def fast_atan_numpy(x):
    """Fast arctangent approximation using NumPy."""
    x = np.float32(x)
    atan = x / (1.0 + 0.28 * x * x)
    return atan


def fast_cosh(x):
    """Fast hyperbolic cosine approximation using NumPy.
    Uses truncated Taylor series expansions for cosh
    """
    x = np.float32(x)
    x2 = x * x
    cosh = 1 + x2 / 2 + x2 * x2 / 24
    return cosh


def fast_sinh(x):
    """Fast hyperbolic sine approximation using NumPy.
    Uses truncated Taylor series expansions for sinh
    """
    x = np.float32(x)
    x2 = x * x
    sinh = x * (1 + x2 / 6 + x2 * x2 / 120)
    return sinh


def fast_log_small(x):
    """Fast natural logarithm approximation for small x using NumPy.
    Uses a truncated Taylor series expansion around x = 1.
    """
    x = np.float32(x)
    y = x - 1
    log = y - y * y / 2 + y * y * y / 3
    return log


def fast_exp_small(x):
    """Fast exponential approximation for small x using NumPy.
    Uses a truncated Taylor series expansion around x = 0.
    """
    x = np.float32(x)
    exp = 1 + x + x * x / 2 + x * x * x / 6
    return exp


def fast_erf(x):
    """Fast error function approximation using NumPy."""
    # Abramowitz and Stegun approximation
    a1 = 0.278393
    a2 = 0.230389
    a3 = 0.000972
    a4 = 0.078108
    x = np.float32(x)
    sign = np.sign(x)
    x = np.abs(x)
    t = 1 / (1 + a1 * x + a2 * x * x + a3 * x**3 + a4 * x**4)
    erf = sign * (1 - t**4)
    return erf


def fast_digamma(x):
    """Fast digamma function approximation using NumPy."""
    x = np.float32(x)
    digamma = np.log(x) - 1 / (2 * x)
    return digamma


def fast_lgamma(x):
    """Fast log-gamma function approximation using NumPy.
    Uses the logarithm of Stirling's approximation.

    """
    x = np.float32(x)
    x_minus_one = x - 1
    lgamma = (
        (x_minus_one + 0.5) * np.log(x_minus_one) - x_minus_one + 0.9189385332
    )  # 0.918... = 0.5 * ln(2π)
    return lgamma


def fast_gamma(x):
    """Fast gamma function approximation using NumPy.
    Uses Stirling's approximation for the gamma function.
    """
    x = np.float32(x)
    sqrt_two_pi = np.float32(2.5066282746310002)
    x_minus_one = x - 1
    gamma = sqrt_two_pi * x_minus_one ** (x_minus_one + 0.5) * np.exp(-x_minus_one)
    return gamma


# import numpy as np
# import timeit

# # Fast inverse square root
# number = np.array([1.0, 2.0, 4.0, 9.0], dtype=np.float32)
# inv_sqrt = Q_rsqrt_numpy(number)
# print("Fast inverse square roots:", inv_sqrt)
# print("Standard inverse square roots:", 1 / np.sqrt(number))

# %timeit Q_rsqrt_numpy(number)
# %timeit 1 / np.sqrt(number)

# # Fast sine
# angles = np.linspace(0, 2 * np.pi, 5, dtype=np.float32)
# fast_sines = fast_sin_numpy(angles)
# fast_sines = fast_sin_lut(angles, 10000000)
# print("Fast sines:", fast_sines)
# print("Standard sines:", np.sin(angles))

# %timeit fast_sin_numpy(angles)
# %timeit fast_sin_numpy_fifth(angles)
# %timeit np.sin(angles)

# values = np.array([1.0, 2.0, 4.0, 10.0], dtype=np.float32)
# approx_log = fast_log_numpy(values)
# standard_log = np.log(values)

# print("Approximate log:", approx_log)
# print("Standard log:", standard_log)

# abs_error = np.abs(approx_log - standard_log)
# relative_error = abs_error / standard_log
# print("Absolute error:", abs_error)
# print("Relative error:", relative_error)

# %timeit fast_log_numpy(values)
# %timeit np.log(values)


# values = np.array([1.0, 2.0, 4.0, 0.5], dtype=np.float32)
# approx_inv = fast_inverse_numpy(values)
# standard_inv = 1 / values

# print("Approximate inverse:", approx_inv)
# print("Standard inverse:", standard_inv)


# %timeit fast_inverse_numpy(values)
# %timeit 1/ values

# values = np.array([1, 2, 4, 9, 16, 25, 36, 49, 64, 81], dtype=np.uint32)
# approx_isqrt = fast_isqrt_numpy(values)
# standard_isqrt = np.floor(np.sqrt(values)).astype(np.uint32)

# print("Approximate integer sqrt:", approx_isqrt)
# print("Standard integer sqrt:", standard_isqrt)


# %timeit fast_isqrt_numpy(values)
# %timeit np.floor(np.sqrt(values))


# values = np.array([1.0, 8.0, 27.0, 64.0], dtype=np.float32)
# approx_cbrt = fast_cbrt_numpy(values)
# standard_cbrt = np.cbrt(values)

# print("Approximate cube root:", approx_cbrt)
# print("Standard cube root:", standard_cbrt)


# %timeit fast_cbrt_numpy(values)
# %timeit np.cbrt(values)


# angles = np.linspace(-np.pi / 2 + 0.1, np.pi / 2 - 0.1, 10, dtype=np.float32)
# approx_tan = fast_tan_numpy(angles)
# standard_tan = np.tan(angles)

# print("Approximate tangent:", approx_tan)
# print("Standard tangent:", standard_tan)


# %timeit fast_tan_numpy(values)
# %timeit np.tan(values)


# values = np.array([0.0, 1.0, 2.0, -1.0], dtype=np.float32)
# approx_exp2 = fast_exp2_numpy(values)
# standard_exp2 = 2 ** values

# print("Approximate exp2:", approx_exp2)
# print("Standard exp2:", standard_exp2)


# %timeit fast_exp2_numpy(values)
# %timeit 2 ** values


# values = np.array([1.0, 2.0, 4.0, 8.0], dtype=np.float32)
# approx_log2 = fast_log2_numpy(values)
# standard_log2 = np.log2(values)

# print("Approximate log2:", approx_log2)
# print("Standard log2:", standard_log2)


# %timeit fast_log2_numpy(values)
# %timeit np.log2(values)


# values = np.linspace(-1.0, 1.0, 10, dtype=np.float32)
# approx_atan = fast_atan_numpy(values)
# standard_atan = np.arctan(values)

# print("Approximate arctangent:", approx_atan)
# print("Standard arctangent:", standard_atan)

# %timeit fast_atan_numpy(values)
# %timeit np.arctan(values)


# values = np.linspace(-2, 2, 10, dtype=np.float32)
# approx_sinh = fast_sinh(values)
# standard_sinh = np.sinh(values)

# approx_cosh = fast_cosh(values)
# standard_cosh = np.cosh(values)

# print("Approximate sinh:", approx_sinh)
# print("Standard sinh:", standard_sinh)
# print("Approximate cosh:", approx_cosh)
# print("Standard cosh:", standard_cosh)

# %timeit fast_sinh(values)
# %timeit np.sinh(values)
# %timeit fast_cosh(values)
# %timeit np.cosh(values)


# values = np.linspace(0.9, 1.1, 100, dtype=np.float32)
# approx_log = fast_log_small(values)
# standard_log = np.log(values)

# print("Approximate log (small x):", approx_log)
# print("Standard log:", standard_log)
# %timeit fast_log_small(values)
# %timeit np.log(values)


# values = np.linspace(-0.5, 0.5, 10, dtype=np.float32)
# approx_exp = fast_exp_small(values)
# standard_exp = np.exp(values)

# print("Approximate exp (small x):", approx_exp)
# print("Standard exp:", standard_exp)
# %timeit fast_exp_small(values)
# %timeit np.exp(values)

# abs_error = np.abs(approx_exp - standard_exp)
# max_abs_error = np.max(abs_error)
# print("Maximum absolute error:", max_abs_error)

# base = np.array([2.0, 3.0, 4.0, 5.0], dtype=np.float32)
# exponent = np.array([2.0, 1.5, 0.5, -1.0], dtype=np.float32)
# approx_pow = fast_pow(base, exponent)
# standard_pow = np.power(base, exponent)

# print("Approximate power:", approx_pow)
# print("Standard power:", standard_pow)
# %timeit fast_pow(base, exponent)
# %timeit np.power(base, exponent)

# from scipy.special import erf
# values = np.linspace(-3, 3, 10, dtype=np.float32)
# approx_erf = fast_erf(values)
# standard_erf = erf(values)  # Import erf from scipy.special

# print("Approximate erf:", approx_erf)
# print("Standard erf:", standard_erf)
# %timeit fast_erf(values)
# %timeit erf(values)

# values = np.linspace(1, 5, 10, dtype=np.float32)
# approx_gamma = fast_gamma(values)
# from scipy.special import gamma
# standard_gamma = gamma(values)

# print("Approximate gamma:", approx_gamma)
# print("Standard gamma:", standard_gamma)
# %timeit fast_gamma(values)
# %timeit gamma(values)


# values = np.linspace(1, 10, 10, dtype=np.float32)
# approx_lgamma = fast_lgamma(values)
# from scipy.special import gammaln
# standard_lgamma = gammaln(values)

# print("Approximate lgamma:", approx_lgamma)
# print("Standard lgamma:", standard_lgamma)
# %timeit fast_lgamma(values)
# %timeit gammaln(values)


# values = np.linspace(1, 10, 10, dtype=np.float32)
# approx_digamma = fast_digamma(values)
# from scipy.special import psi
# standard_digamma = psi(values)

# print("Approximate digamma:", approx_digamma)
# print("Standard digamma:", standard_digamma)
# %timeit fast_digamma(values)
# %timeit psi(values)


# def fast_j0(x):
#     """
#     Fast Bessel function J0 approximation using NumPy.
#     TODO: needs work
#     """
#     x = np.float32(x)
#     y = 1 - x * x / 4 + x ** 4 / 64
#     return y


# def fast_j1(x):
#     """
#     Fast Bessel function J1 approximation using NumPy.
#     TODO: needs work
#     """
#     x = np.float32(x)
#     y = x / 2 - x ** 3 / 16 + x ** 5 / 384
#     return y


# values = np.linspace(0, 5, 10, dtype=np.float32)
# approx_j0 = fast_j0(values)
# from scipy.special import j0
# standard_j0 = j0(values)

# approx_j1 = fast_j1(values)
# from scipy.special import j1
# standard_j1 = j1(values)

# print("Approximate J0:", approx_j0)
# print("Standard J0:", standard_j0)
# print("Approximate J1:", approx_j1)
# print("Standard J1:", standard_j1)


# import numpy as np

# def traditional_softmax(x):
#     """
#     Compute the softmax of each row of the input x.
#     """
#     x_exp = np.exp(x)
#     x_sum = np.sum(x_exp, axis=-1, keepdims=True)
#     return x_exp / x_sum


# def optimized_softmax(x):
#     """
#     Optimized softmax function with numerical stability.
#     """
#     # Subtract the max for numerical stability
#     x_max = np.max(x, axis=-1, keepdims=True)
#     x_sub = x - x_max

#     # Use exp approximation for faster computation
#     # Approximate exp(x) using a bit-level trick
#     x_exp = fast_exp_numpy(x_sub)

#     x_sum = np.sum(x_exp, axis=-1, keepdims=True)
#     return x_exp / x_sum


# x = np.random.rand(1000, 512).astype(np.float32)

# softmax_traditional = traditional_softmax(x)
# softmax_optimized = optimized_softmax(x)

# print("Approximate softmax:", softmax_optimized)
# print("Standard softmax:", softmax_traditional)

# difference = np.max(np.abs(softmax_traditional - softmax_optimized))
# print(f"Maximum difference: {difference}")
# ## -> Maximum difference: 6.984919309616089e-10


# %timeit traditional_softmax(x)

# %timeit optimized_softmax(x)
