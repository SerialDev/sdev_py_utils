class ch_utils(object):
    @staticmethod
    def create_client(_ip="0.0.0.0", port="9000"):
        from clickhouse_driver import Client

        client = Client(
            _ip,
            port=port,
            # user='python',
            # password='secret',
            # database='marketing',
            secure=False,
            verify=False,
            compression=False,
        )
        return client

    @staticmethod
    def ch_df_map(inverse=False):
        import numpy as np

        data_mapper = {
            "DateTime64 CODEC(DoubleDelta)": np.datetime64,
            "DateTime CODEC(DoubleDelta)": np.datetime64,
            "Float64 CODEC(LZ4)": np.float64,
            "Float32 CODEC(LZ4)": np.float32,
            "Float16 CODEC(LZ4)": np.float16,
            "Int64 CODEC(LZ4)": np.int64,
            "Int32 CODEC(LZ4)": np.int32,
            "Int16 CODEC(LZ4)": np.int16,
            "Int8 CODEC(LZ4)": np.int8,
            "UInt8 CODEC(LZ4)": np.uint8,
            "UInt16 CODEC(LZ4)": np.uint16,
            "UInt32 CODEC(LZ4)": np.uint32,
            "UInt64 CODEC(LZ4)": np.uint64,
            "String CODEC(LowCardinalitty)": np.object,
            "IPv6 CODEC(LZ4)": np.object,
            "String CODEC(LZ4)": np.object_,
            "UInt8 CODEC(LZ4)": np.bool_,
        }
        if inverse == True:
            inv_map = {}
            for k, v in data_mapper.items():
                inv_map[v] = inv_map.get(v, []) + [k]
            data_mapper = inv_map
        else:
            data_mapper = {k.split(" ")[0]: v for k, v in data_mapper.items()}
        return data_mapper

    @staticmethod
    def create_user(client, _name="user", pw=""):
        return client.execute(
            f"CREATE USER IF NOT EXISTS {_name} IDENTIFIED WITH sha256_password BY '{pw}'"
        )

    @staticmethod
    def create_db(client, name):
        # TODO: Add on-cluster
        return client.execute(f"CREATE DATABASE IF NOT EXISTS {name} ")

    @staticmethod
    def use_db(client, name):
        # TODO: Add on-cluster
        return client.execute(f"USE {name} ")

    @staticmethod
    def show_dbs(client, _like=None):
        if _like is None:
            return client.execute("SHOW databases")
        else:
            return client.execute(f"SHOW databases ILIKE {_like}")

    @staticmethod
    def show_processes(client):
        return client.execute("SHOW processlist")

    @staticmethod
    def show_tables(client, _from="system"):
        return client.execute(f"SHOW tables from {_from}")

    @staticmethod
    def show_dicts(client, _from="system"):
        return client.execute(f"SHOW dictionaries from {_from}")

    @staticmethod
    def show_grants(client, _for="user"):
        return client.execute(f"SHOW GRANTS for {_for}")

    @staticmethod
    def show_users(client):
        return client.execute(f"SHOW USERS")

    @staticmethod
    def show_roles(client, current=False):
        if current:
            return client.execute(f"SHOW CURRENT ROLES")
        else:
            return client.execute(f"SHOW ENABLED ROLES")

    @staticmethod
    def show_user(client, _name="default"):
        return client.execute(f"SHOW CREATE USER {_name}")

    @staticmethod
    def show_profiles(client):
        return client.execute(f"SHOW SETTINGS PROFILES")

    @staticmethod
    def show_quotas(client):
        return client.execute(f"SHOW QUOTAS")

    @staticmethod
    def show_datatypes(client):
        #  WHERE alias_to = 'String'
        return client.execute(f"SELECT * FROM system.data_type_families")

    @staticmethod
    def show_clusters(client):
        return client.execute(f"SELECT * FROM system.clusters FORMAT Vertical;")

    @staticmethod
    def pd_col_types(df):
        import numpy as np

        data_types = list(zip(df.columns, df.dtypes))

        data_mapper = ch_utils().ch_df_map(True)

        data_types = np.array([[x[0], data_mapper[x[1].type]] for x in data_types])

        return data_types

    @staticmethod
    def pd_execute(ch, client, query):
        import pandas as pd

        a = client.execute(query, with_column_types=True)
        datatypes = []
        columns = []
        mapping = ch_utils().ch_df_map()
        for data, datatype in a[1]:
            try:
                columns.append(data)
                datatypes.append(mapping[datatype.split("(")[0]])
            except Exception:
                pass
        # dtypes = list(zip(columns,datatypes))
        # record = np.array(map(tuple, a[0]), dtype=dtypes)
        # mydf = pd.DataFrame.from_records(record)
        return pd.DataFrame(a[0], columns=columns)

    @staticmethod
    def enclose(data: str) -> str:
        return f"({data})"

    @staticmethod
    def enclose_quote(data: str) -> str:
        return f"'{data}'"

    @staticmethod
    def create_table(
        ch, df, db_name, table_name, primary_key="date", low_cardinality=False
    ):
        data_types = ch.pd_col_types(df)
        if low_cardinality:
            data_types = [
                [
                    k,
                    v[0].replace(
                        "String CODEC(LZ4)", "LowCardinality(String) CODEC(LZ4)"
                    ),
                ]
                for k, v in data_types
            ]
        else:
            data_types = [[k, v[0]] for k, v in data_types]

        query = f"""create table if not exists {db_name}.{table_name}
        {ch.enclose(", ".join([" ".join(x) for x in data_types]))}
        ENGINE = MergeTree()
        PRIMARY KEY ({primary_key})"""
        return query

    @staticmethod
    def enclose_columns(ch, df, col_list):
        for col in col_list:
            df[f"{col}"] = df[f"{col}"].apply(ch.enclose_quote)
        return df

    @staticmethod
    def escape_columns(ch, df, col_list):
        for col in col_list:
            df[f"{col}"] = df[f"{col}"].apply(lambda x: x.replace("'", ""))
        return df

    @staticmethod
    def drop_table(client, db_name, table_name):
        return client.execute(f"DROP TABLE {db_name}.{table_name};")

    @staticmethod
    def count_rows_table(client, db_name, table_name):
        return client.execute(f"select count(*) from {db_name}.{table_name};")

    @staticmethod
    def insert_data(ch, df, db_name, table_name):
        data = ", ".join(
            [ch.enclose(", ".join(list(map(str, x)))) for idx, x in df.iterrows()]
        )
        query = f"""
        INSERT INTO {db_name}.{table_name}
        (*)
        VALUES
        {data}
        ;
        """
        return query


def enclose(data: str) -> str:
    return f"({data})"


def enclose_quote(data: str) -> str:
    return f"'{data}'"


from subprocess import Popen, PIPE, DEVNULL
import pandas as pd
import os
from timeit import default_timer as timer
from string import Template
from requests import post
from datetime import date, timedelta
import numpy as np
from tqdm import tqdm
import time

# CLICKHOUSE_HOST
def access_login(host):
    p = Popen(
        ["cloudflared", "access", "login", CLICKHOUSE_HOST],
        stdout=PIPE,
        stderr=DEVNULL,
        encoding="UTF-8",
        universal_newlines=True,
    )
    p.wait()
    return p


_macros = {
    "zoneName": "dictGetString('zone', 'zone_name', toUInt64(zoneId))",
    "userId": "dictGetUInt32('zone', 'user_id', toUInt64(zoneId))",
    "colo": "dictGetString('colo', 'code', toUInt64(edgeColoId))",
    "upperTierColo": "dictGetString('colo', 'code', toUInt64(upperTierColoId))",
    "countryName": "dictGetString('country', 'name', toUInt64(clientCountry))",
    "countryCode2": "dictGetString('country', 'alpha2', toUInt64(clientCountry))",
    "countryCode3": "dictGetString('country', 'alpha3', toUInt64(clientCountry))",
    "method": "dictGetString('method', 'name', toUInt64(clientRequestHTTPMethod))",
    "video": "replaceRegexpAll(UUIDNumToString(videoId), '-', '')",
    "impression": "replaceRegexpAll(UUIDNumToString(impressionId), '-', '')",
    "today": "date=today() and datetime>toDateTime(today())",
    "yesterday": "date=today()-1 and datetime>=toDateTime(today()-1) and datetime<toDateTime(today())",
    "mime": "dictGetString('mime',  'name',  toUInt64(edgeResponseContentType))",
    "startMs": "bitShiftRight(rayId, 28) % 100 * 10",
    "cacheStatus": "dictGetString('cache_status', 'name', toUInt64(cacheStatus))",
}


def _preprocess_query(q):
    return Template(q).substitute(_macros) + "\nFORMAT TabSeparatedWithNames"


def _execute_query(q, host, **kwargs):
    """kwargs can define the table name, data, and schema like:

    table_name=(dataframe, 'col1 Type1, col2 Type2')
    """
    files = {}
    params = {}
    post_body = ""

    if kwargs:
        # If we're going to upload a file (i.e. kwargs are present), we need to
        # URL-encode the query and include it in the URL
        params["query"] = q
    else:
        # Otherwise it's safer to include it as a POST body
        post_body = q

    for name, v in kwargs.items():
        # Table schema is always included in the URL
        params[name + "_structure"] = v[1]
        table_data = v[0]

        # Do some type inference and try to encode correctly
        if type(table_data) is pd.core.frame.DataFrame:
            files[name] = table_data.to_csv(sep="\t", index=False, header=False)
        elif type(table_data) is str:
            # Strings must be pre-formatted as tsv
            files[name] = (name, table_data)
        elif type(table_data[0]) is str:
            # List of strings can be joined
            files[name] = (name, "\n".join(table_data))
        else:
            # List of lists can also be joined into TSV
            files[name] = (name, "\n".join("\t".join(map(str, x)) for x in table_data))

    if not str(host).startswith("https://"):
        host = "https://%s" % host

    try:
        auth_token = (
            Popen(
                ["cloudflared", "access", "token", "-app=" + host],
                stdout=PIPE,
                stderr=DEVNULL,
                encoding="UTF-8",
                universal_newlines=True,
            )
            .communicate()[0]
            .splitlines()[0]
            .rstrip()
        )
    except IndexError:
        print(
            """
There was an error fetching your CF access token.
Make sure you have authenticated with CF Access first by running a command like:
`%login clickhouse-pdx-root.bi.cfdata.org`
"""
        )
        raise

    response = post(
        host,
        data=post_body,
        params=params,
        files=files,
        stream=True,
        auth=(CLICKHOUSE_USER, CLICKHOUSE_PW),
        cookies={"CF_Authorization": auth_token},
    )
    return response


def query(q, host, silent=False, **kwargs):
    """Main function for running clickhouse queries.

    This is meant to be called primarily with the %%query cell magic. However it can also be called
    directly as clickhouse.query() with kwargs if you want to pass an external table, e.g.:

    clickhouse.query('''
        SELECT ...
        FROM requests_sample
        WHERE date >= yesterday() - 1
        AND clientIPv4 global in (select IPv4StringToNum(ip) from ip_table)
        AND ...
        ''', ip_table=(ips[['ip4']].values, 'ip String'))
    """
    start = timer()
    full_q = _preprocess_query(q)
    r = _execute_query(full_q, host, **kwargs)
    end = timer()
    if not r.ok:
        # Probably a query parse error returned from Clickhouse
        if "DB::Exception" in r.text:
            print("Host: %s\n\n%s\n%s" % (host, r.text, full_q))
            return
        else:
            raise RuntimeError(r.text)
    if not silent:
        print("Query finished in %.02fs" % (end - start))
    return pd.read_csv(r.raw, sep="\t", parse_dates=True)


def describe_table(tbl, host):
    """ Returns dataframe with schema for a given table """
    p = tbl.split(".")
    if len(p) != 2:
        db = "default"
    else:
        (db, tbl) = p
    q = (
        "select name, type, default_expression from system.columns \
         where database = '%s' and table = '%s'"
        % (db, tbl)
    )
    return query(q, host=host)


def query_date_range(
    query_text, start=None, end=date.today(), days=None, dfs=[], **kwargs
):
    """Runs `query` over multiple days from `start` to `end` and concatenates
    into one dataframe.

    If you include a column 'd' in your query output that represents a day,
    it will add a nice new 'date' column as an index
    """

    day = timedelta(days=1)

    if type(query_text) is str:
        q = Template(query_text)
    elif type(query_text) is Template:
        q = query_text
    else:
        raise ValueError("Need a string or template for query")

    # Replace "daterange" macro if it exists
    q = Template(
        q.substitute(
            {
                "daterange": "date = '$date1' \
        and datetime >= toDateTime('$date1 00:00:00') \
        and datetime <  toDateTime('$date2 00:00:00')"
            }
        )
    )

    # Get start date
    if type(days) is int:
        start_date = date.today() - timedelta(days=days)
    elif type(start) is str:
        start_date = parser.parse(start).date()
    elif type(start) is date:
        start_date = start
    else:
        raise ValueError("Need a start_date as a string or date, or a number of days")

    # Get end date
    if type(end) is str:
        end_date = parser.parse(end).date()
    elif type(end) is date:
        end_date = end
    else:
        raise ValueError("Need an end_date as a string or a date (or None)")

    d = start_date
    while d < end_date:
        date1 = d.isoformat()
        date2 = (d + day).isoformat()
        print(date1, date2)
        dfs.append(query(q.substitute({"date1": date1, "date2": date2}), **kwargs))
        d += day

    df = pd.concat(dfs, sort=False)
    if "d" in df:
        df["date"] = pd.to_datetime(df.d)
        df.set_index("date")
    return df
