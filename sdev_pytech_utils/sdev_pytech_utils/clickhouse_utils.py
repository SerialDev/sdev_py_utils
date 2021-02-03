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
        data_types = list(zip(df.columns, df.dtypes))

        data_mapper = ch_utils().ch_df_map(True)

        data_types = np.array([[x[0], data_mapper[x[1].type]] for x in data_types])

        return data_types

    @staticmethod
    def pd_execute(query):
        import pandas as pd

        a = client.execute(query, with_column_types=True)
        datatypes = []
        columns = []
        mapping = ch.ch_df_map()
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
