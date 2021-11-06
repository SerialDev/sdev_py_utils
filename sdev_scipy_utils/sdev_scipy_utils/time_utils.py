#                    Encoding Time into cyclical features                   #
# ------------------------------------------------------------------------- #


def sin_cos(n):
    from math import sin, cos, pi

    theta = 2 * pi * n
    return (sin(theta), cos(theta))


def get_cycles(d, multi_str=""):
    """
    Get the cyclic properties of a datetime,
    represented as points on the unit circle.
    Arguments
    ---------
    d : datetime object
    Returns
    -------
    dictionary of sine and cosine tuples
    """
    from datetime import datetime

    days_in_month = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    month = d.month - 1
    day = d.day - 1
    month_sc = sin_cos(month / 12)
    day_sc = sin_cos(day / days_in_month[month])
    weekday_sc = sin_cos(d.weekday() / 7)
    hour_sc = sin_cos(d.hour / 24)
    minute_sc = sin_cos(d.minute / 60)
    second_sc = sin_cos(d.second / 60)
    return {
        f"month_sin{multi_str}": month_sc[0],
        # f"month_cos{multi_str}": month_sc[1],
        f"day_sin{multi_str}": day_sc[0],
        # f"day_cos{multi_str}": day_sc[1],
        f"weekday_sin{multi_str}": weekday_sc[0],
        # f"weekday_cos{multi_str}": weekday_sc[1],
        f"hour_sin{multi_str}": hour_sc[0],
        # f"hour_cos{multi_str}": hour_sc[1],
        f"minute_sin{multi_str}": minute_sc[0],
        # f"minute_cos{multi_str}": minute_sc[1],
        f"second_sin{multi_str}": second_sc[0],
        # f"second_cos{multi_str}": second_sc[0],
    }


def pd_timestamp_to_cycles(df, col_name, multi_str=""):
    temp = pd.DataFrame(
        df[col_name].apply(lambda x: get_cycles(x, multi_str=multi_str)).to_list()
    )
    df = pd.concat([df, temp], axis=1)
    return df


def pd_list_to_pd(result, batches):
    from math import ceil

    chunk_len = int(ceil(len(result)) / batches)
    final_df_result = []
    for batch in range(batches + 1):
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


def timeline_label_datapoint(ordinal_string, start_time, end_time, extra=None):
    if extra:
        return {
            "timeRange": [start_time, end_time],
            "val": ordinal_string,
            "extra": extra,
        }
    return {"timeRange": [start_time, end_time], "val": ordinal_string}


def timeline_group_datapoint(label_name, label_datapoints):
    return {"label": label_name, "data": label_datapoints}


def timeline_chart_datapoint(group_name, group_datapoints):
    return {"group": group_name, "data": group_datapoints}


def pd_table_fmt(df):
    df = df.fillna("-")
    cols = list(df.columns)
    contents = df.to_numpy()
    return [cols, [list(i) for i in contents]]


def pd_cols(df):
    if type(df) == pd.core.frame.DataFrame:
        cols = list(df.columns)
    elif type(df) == pd.core.series.Series:
        cols = list(df.index)
    return cols


def cache_feature_encoder(feature_col, filename=None):
    import sklearn as sk

    # returns: an instance of sklearn OneHotEncoder fit against a (training) column feature;
    # such instance is saved and can then be loaded to transform unseen data
    enc = sk.preprocessing.OneHotEncoder(
        handle_unknown="ignore"
    )  # Allow for diff categories in real datasets

    feature_vec = feature_col.sort_values().values.reshape(-1, 1)
    enc.fit(feature_vec)
    # feature_vec = feature_col.sort_values().values.reshape(1, -1)
    # enc.fit(feature_vec)

    feature_cols = enc.categories_
    buf = to_buffer([enc, feature_cols])
    if filename is not None:
        with open(filename, "wb") as f:
            f.write(buf)
    return buf


def pd_encode_cached_encoder(
    feature_col, filename=None, feature_encoder=None, full_df=None
):
    # maps an unseen column feature using one-hot-encoding previously fit against training data
    # returns: a pd.DataFrame of newly one-hot-encoded feature
    if filename is not None:
        with open(filename, "rb") as f:
            feature_encoder = f.read()
    if feature_encoder is not None:
        enc, feature_cols = from_buffer(feature_encoder)
        unseen_vec = feature_col.values.reshape(-1, 1)
        encoded_vec = enc.transform(unseen_vec).toarray()
        encoded_df = pd.DataFrame(encoded_vec, columns=feature_cols)
        if full_df is not None:
            full_df[enc.categories_[0]] = encoded_df
            return full_df
        return encoded_df
    raise Exception("filename or feature_encoder must be provided")


class Batcher:
    # Usage
    # net = net.train()  # explicitly set
    # bat_size = 40
    # loss_func = T.nn.MSELoss()
    # optimizer = T.optim.Adam(net.parameters(), lr=0.01)
    # batcher = Batcher(num_items=len(norm_x),
    #   batch_size=bat_size, seed=1)
    # max_epochs = 100
    def __init__(self, num_items, batch_size, seed=0):
        self.indices = np.arange(num_items)
        self.num_items = num_items
        self.batch_size = batch_size
        self.rnd = np.random.RandomState(seed)
        self.rnd.shuffle(self.indices)
        self.ptr = 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.ptr + self.batch_size > self.num_items:
            self.rnd.shuffle(self.indices)
            self.ptr = 0
            raise StopIteration  # ugh.
        else:
            result = self.indices[self.ptr : self.ptr + self.batch_size]
            self.ptr += self.batch_size
            return result


def flask_print(stderr=[], stdout=[]):
    import sys

    for i in stderr:
        print(i, file=sys.stderr)
    for i in stdout:
        print(i, file=sys.stdout)
    sys.stdout.flush()
    sys.stderr.flush()


def gen_query_string(var_name, content):
    return f"?{var_name}={content}"


def max_len_aos(data):
    max_len = (0, 0)
    for idx, i in enumerate(data):
        try:
            current_len = len(i.keys())
        except Exception:
            current_len = 0
        if current_len > max_len[1]:
            max_len = (idx, current_len)
    return max_len


def mdeconstruct(dict_list, key_list):
    query = "lambda x: ["
    for current_row in key_list:
        if type(current_row) == list:
            for idx, i in enumerate(current_row):
                if idx == 0:
                    query += f"x['{i}']"
                else:
                    query += f"['{i}']"
            query += ", "
        else:
            query += f"x['{current_row}']"
            query += ", "
    query += "]"
    print(query)
    return list(map(eval(query), dict_list))


#                                Encoding IP                                #
# ------------------------------------------------------------------------- #


def geo_lookup_ip(ip):
    from geolite2 import geolite2

    reader = geolite2.reader()
    geo_reader = reader.get(ip)
    result = {}
    try:
        result["subdivisions"] = geo_reader["subdivisions"][0]["iso_code"]
    except Exception:
        result["subdivisions"] = "NA"
    try:
        result["continent"] = geo_reader["continent"]["code"]
    except Exception:
        result["continent"] = "NA"
    try:
        result["country"] = geo_reader["country"]["iso_code"]
    except Exception:
        result["country"] = "NA"
    try:
        result["city"] = geo_reader["city"]["names"]["de"]
    except Exception:
        result["city"] = "NA"
    try:
        result["postal"] = geo_reader["postal"]["code"]
    except Exception:
        result["postal"] = "NA"
    return result


def pd_geo_lookup(df, col_name):
    temp = pd.DataFrame(df[col_name].apply(lambda x: geo_lookup_ip(x)).to_list())
    df = pd.concat([df, temp], axis=1)
    return df


def pd_h_concat(df_1, df_2, cols=None, transform=False, keep_col=False):
    if cols is not None:
        df_2 = df_2[cols]
        if transform == True and type(cols) == str:
            df_2 = transform_aos_pd(df_2)
    df = pd.concat([df_1, df_2], axis=1, verify_integrity=True)
    if cols is not None and keep_col == False:
        df.drop(cols, axis=1, inplace=True)
    return df


def pct_col_nul(df, col):
    print(
        f"""Dataframe col {col} Nulls accounts for 
          {(df.shape[0] - df[col].dropna().shape[0]) * 100 / df.shape[0]:.2f}% 
          of the whole DataFrame of len={df.shape[0]}"""
    )


def split_keep(s, delimiter):
    split = s.split(delimiter)
    return [substr + delimiter for substr in split[:-1]] + [split[-1]]


def depth_flatten(array, depth=2):
    result = array
    for i in range(depth):
        result = functools.reduce(operator.iconcat, result, [])
    return result


def parse_uri(s):
    return [
        [[split_keep(z, "=") for z in split_keep(y, "&")] for y in split_keep(x, "?")]
        for x in s.split("/")
    ]


def get_by_path(root, items):
    """Access a nested object in root by item sequence."""
    from functools import reduce  # forward compatibility for Python 3
    import operator

    return reduce(operator.getitem, items, root)


def set_by_path(root, items, value):
    """Set a value in a nested object in root by item sequence."""
    get_by_path(root, items[:-1])[items[-1]] = value


def del_by_path(root, items):
    """Delete a key-value in a nested object in root by item sequence."""
    del get_by_path(root, items[:-1])[items[-1]]


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


def flatten_dictionary(d, parent_key="", sep="_"):
    import collections

    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.MutableMapping):
            items.extend(flatten_dictionary(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def type_executor(tuple_list, data_param):
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


def flatten_dict(current_dict, key=None):

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


def print_return(x, key=""):
    if key == None:
        key = ""
    # print(x)
    return key + "_" + x


def identity(x):
    return x


def deconstruct(dict_list, key):
    return list(map(lambda x: x[key], dict_list))


def uniquify_list(seq):  # Dave Kirby
    # Order preserving
    seen = set()
    return [x for x in seq if x not in seen and not seen.add(x)]


def empty_or_nan(data):
    try:
        return data[0]
    except Exception:
        return np.nan


def check_dict(dict_, key):
    try:
        return dict_[key]
    except Exception:
        return np.nan


def mdeconstruct(dict_list, key_list):
    query = "lambda x: ["
    for current_row in key_list:
        if type(current_row) == list:
            for idx, i in enumerate(current_row):
                if idx == 0:
                    query += f"x['{i}']"
                else:
                    query += f"['{i}']"
            query += ", "
        else:
            query += f"x['{current_row}']"
            query += ", "
    query += "]"
    print(query)
    return list(map(eval(query), dict_list))
