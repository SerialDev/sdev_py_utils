"""Python Postgres utility library"""

import sqlalchemy
import pandas as pd


class pg(object):
    """
    Postgresql sqlalchemy based utililities

    Parameters
    ----------

    get_engine : ip|str port|str
       Get an engine object for postgres

    show_tables : engine|pyObj
       show the tables existing in db {engine} is connected to

    show_running_queries : engine|pyObj
       Show queries currently running

    kill_running_query : engine|pyObj procpid|str
       Kill a running query based on its id

    kill_idle_query : engine|pyObj procpid|str
       Kill an idle query based on its id

    show_database_sizes : engine|pyObj|Show the size of database
       nil

    show_tables_and_views_usage : engine|pyObj
       show usage by tables and by views

    show_long_running : engine|pyObj
       show all long running queries

    count_indexes : engine|pyObj
       Check how many indexes are in cache

    get_db_encoding : engine|pyObj db_name|str
       Get the string encoding of a db

    get_locks_info_pd : engine|pyObj
       Get information on what locks are currently active

    get_db_indexes_pd : engine|pyObj
       nil

    db_name|str :  get database indexes currently in use
       nil

    get_table_sizes : engine|pyObj
       Get a list of all table sizes in connected db

    get_cache_hit_pd : engine|pyObj
       Get information on cache hits and cache misses for each table

    get_index_usage_pd : engine|pyObj
       Get information on index usage for a given db

    get_index_cache_hit_pd : engine|pyObj
       Get information on index usage that is hitting a good cache locality

    drop_table : engine|pyObj table_name|str
       Drop a table from db

    non_seq_query : engine|pyObj query|str mem|str
       Do a Non-sequential scan with a given working memory, use to override defaults
       if they are slow

    to_sql : df|pd.DataFrame engine|pyObj table|str if_exists|str sep|str encoding|str
       Highly efficient write to postgresql from pandas

    """

    @staticmethod
    def get_engine(ip, port, user="postgres", pwd="postgres"):
        engine = sqlalchemy.create_engine(
            f"postgresql+psycopg2://{user}:{pwd}@{ip}:{port}"
        )
        return engine

    @staticmethod
    def show_tables(engine):
        return engine.execute(
            "select * from pg_catalog.pg_tables where schemaname != 'information_schema' and schemaname != 'pg_catalog';"
        ).fetchall()

    @staticmethod
    def show_tables_pd(engine):
        return pd.read_sql(
            "select * from pg_catalog.pg_tables where schemaname != 'information_schema' and schemaname != 'pg_catalog';",
            engine,
        )

    @staticmethod
    def show_running_queries(engine):
        return engine.execute(
            """SELECT pid, age(query_start, clock_timestamp()), usename, query
FROM pg_stat_activity
WHERE query != '<IDLE>' AND query NOT ILIKE 'pg_stat_activity'
ORDER BY query_start desc;"""
        ).fetchall()

    @staticmethod
    def show_running_queries_pd(engine):
        return pd.read_sql(
            """SELECT pid, age(query_start, clock_timestamp()), usename, query
FROM pg_stat_activity
WHERE query != '<IDLE>' AND query NOT ILIKE 'pg_stat_activity'
ORDER BY query_start desc;""",
            engine,
        )

    @staticmethod
    def kill_running_query(engine, procpid):
        return engine.execute(
            """SELECT pg_cancel_backend({procpid})""".format(procpid=procpid)
        )

    @staticmethod
    def kill_idle_query(engine, procpid):
        return engine.execute(
            """SELECT pg_terminate_backend({procpid})""".format(procpid=procpid)
        )

    @staticmethod
    def show_database_sizes(engine):
        return pd.read_sql("SELECT * from pg_user", engine)

    @staticmethod
    def show_database_sizes_pd(engine):
        return pd.read_sql("SELECT * from pg_user", engine)

    @staticmethod
    def show_tables_and_views_usage(engine):
        return engine.execute(
            """with recursive view_tree(parent_schema, parent_obj, child_schema, child_obj, ind, ord) as (select vtu_parent.view_schema, vtu_parent.view_name, vtu_parent.table_schema, vtu_parent.table_name, '', array[row_number() over (order by view_schema, view_name)]from information_schema.view_table_usage vtu_parent where vtu_parent.view_schema = '<SCHEMA NAME>' and vtu_parent.view_name = '<VIEW NAME>' union all select vtu_child.view_schema, vtu_child.view_name, vtu_child.table_schema, vtu_child.table_name, vtu_parent.ind || '  ', vtu_parent.ord || (row_number() over (order by view_schema, view_name))from view_tree vtu_parent, information_schema.view_table_usage vtu_child where vtu_child.view_schema = vtu_parent.child_schema and vtu_child.view_name = vtu_parent.child_obj) select tree.ind || tree.parent_schema || '.' || tree.parent_obj   || ' depends on ' || tree.child_schema || '.' || tree.child_obj txt, tree.ord from view_tree tree order by ord;"""
        ).fetchall()

    @staticmethod
    def show_tables_and_views_usage_pd(engine):
        return pd.read_sql(
            """with recursive view_tree(parent_schema, parent_obj, child_schema, child_obj, ind, ord) as (select vtu_parent.view_schema, vtu_parent.view_name, vtu_parent.table_schema, vtu_parent.table_name, '', array[row_number() over (order by view_schema, view_name)]from information_schema.view_table_usage vtu_parent where vtu_parent.view_schema = '<SCHEMA NAME>' and vtu_parent.view_name = '<VIEW NAME>' union all select vtu_child.view_schema, vtu_child.view_name, vtu_child.table_schema, vtu_child.table_name, vtu_parent.ind || '  ', vtu_parent.ord || (row_number() over (order by view_schema, view_name))from view_tree vtu_parent, information_schema.view_table_usage vtu_child where vtu_child.view_schema = vtu_parent.child_schema and vtu_child.view_name = vtu_parent.child_obj) select tree.ind || tree.parent_schema || '.' || tree.parent_obj   || ' depends on ' || tree.child_schema || '.' || tree.child_obj txt, tree.ord from view_tree tree order by ord;""",
            engine,
        )

    @staticmethod
    def show_long_running(engine):
        return engine.execute(
            """SELECT pid, now() - query_start as "runtime", usename, datname, waiting, state, query FROM  pg_stat_activity WHERE now() - query_start > '2 minutes'::interval and state = 'active' ORDER BY runtime DESC;"""
        ).fetchall()

    @staticmethod
    def pg_seq_scans(engine):
        # -- Sequential Scans
        # -- seq_tup_avg should be < 1000

        return pd.read_sql(
            """
select relname, pg_size_pretty(pg_relation_size(relname::regclass)) as size, seq_scan, seq_tup_read, seq_scan / seq_tup_read as seq_tup_avg from pg_stat_user_tables where seq_tup_read > 0 order by 3,4 desc limit 5;
    """,
            engine,
        )

    @staticmethod
    def count_indexes(engine):
        # -- how many indexes are in cache
        return engine.execute(
            """SELECT sum(idx_blks_read) as idx_read, sum(idx_blks_hit)  as idx_hit, (sum(idx_blks_hit) - sum(idx_blks_read)) / sum(idx_blks_hit) as ratio FROM pg_statio_user_indexes;"""
        ).fetchall()

    @staticmethod
    def pg_check_dirty_pages(engine):
        # -- Dirty Pages
        # -- maxwritten_clean and buffers_backend_fsyn better be = 0
        return pd.read_sql(
            """
        select buffers_clean, maxwritten_clean, buffers_backend_fsync from pg_stat_bgwriter;
        """,
            engine,
        )

    @staticmethod
    def pg_index_cache(engine):
        return pd.read_sql(
            """
SELECT sum(idx_blks_read) as idx_read, sum(idx_blks_hit) as idx_hit, (sum(idx_blks_hit) - sum(idx_blks_read)) / sum(idx_blks_hit) as ratio FROM pg_statio_user_indexes;
    """,
            engine,
        )

    @staticmethod
    def pg_index_usage(engine):
        # -- Index % usage
        return pd.read_sql(
            """
SELECT relname, 100 * idx_scan / (seq_scan + idx_scan) percent_of_times_index_used, n_live_tup rows_in_table FROM pg_stat_user_tables ORDER BY n_live_tup DESC;
    """,
            engine,
        )

    @staticmethod
    def pg_does_table_need_idx(engine):
        # -- Does table needs an Index
        return pd.read_sql(
            """
SELECT relname, seq_scan-idx_scan AS too_much_seq, CASE WHEN seq_scan-idx_scan>0 THEN 'Missing Index?' ELSE 'OK' END, pg_relation_size(relname::regclass) AS rel_size, seq_scan, idx_scan FROM pg_stat_all_tables WHERE schemaname='public' AND pg_relation_size(relname::regclass)>80000 ORDER BY too_much_seq DESC;
    """,
            engine,
        )

    @staticmethod
    def get_db_encoding(engine, db_name):
        t = engine.execute(
            "select pg_encoding_to_char(encoding) from pg_database WHERE datname = '{db_name}';".format(
                db_name=db_name
            )
        )
        return t.fetchall()

    @staticmethod
    def get_locks_info_pd(engine):
        return pd.read_sql(
            "SELECT * FROM pg_locks pl LEFT JOIN pg_stat_activity psa ON pl.pid = psa.pid;",
            engine,
        )

    @staticmethod
    def get_db_indexes_pd(engine, db_name):
        return pd.read_sql(
            """SELECT * FROM pg_indexes WHERE tablename = '{db_name}';""".format(
                db_name=db_name
            ),
            engine,
        )

    @staticmethod
    def get_table_sizes(engine):
        df = pd.read_sql(
            """SELECT
    table_name,
    pg_size_pretty(table_size) AS table_size,
    pg_size_pretty(indexes_size) AS indexes_size,
    pg_size_pretty(total_size) AS total_size
FROM (
    SELECT
        table_name,
        pg_table_size(table_name) AS table_size,
        pg_indexes_size(table_name) AS indexes_size,
        pg_total_relation_size(table_name) AS total_size
    FROM (
        SELECT ('"' || table_schema || '"."' || table_name || '"') AS table_name
        FROM information_schema.tables
    ) AS all_tables
    ORDER BY total_size DESC
) AS pretty_sizes;
        """,
            engine,
        )
        return df

    @staticmethod
    def get_cache_hit_pd(engine):
        return pd.read_sql(
            """
SELECT
  sum(heap_blks_read) as heap_read,
  sum(heap_blks_hit)  as heap_hit,
  sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
FROM
  pg_statio_user_tables;
""",
            engine,
        )

    @staticmethod
    def get_index_usage_pd(engine):
        return pd.read_sql(
            """
SELECT
  relname,
  100 * idx_scan / (seq_scan + idx_scan) percent_of_times_index_used,
  n_live_tup rows_in_table
FROM
  pg_stat_user_tables
WHERE
    seq_scan + idx_scan > 0
ORDER BY
  n_live_tup DESC;
""",
            engine,
        )

    @staticmethod
    def get_index_cache_hit_pd(engine):
        return pd.read_sql(
            """
SELECT
  sum(idx_blks_read) as idx_read,
  sum(idx_blks_hit)  as idx_hit,
  (sum(idx_blks_hit) - sum(idx_blks_read)) / sum(idx_blks_hit) as ratio
FROM
  pg_statio_user_indexes;
""",
            engine,
        )

    @staticmethod
    def drop_table(engine, table_name):
        engine.execute("DROP TABLE {table_name}".format(table_name=table_name))

    @staticmethod
    def non_seq_query(engine, query, mem="6MB"):
        result = engine.execute(
            """BEGIN;
SET LOCAL enable_seqscan= off;
SET LOCAL work_mem = '{mem}';
{query}
COMMIT;""".format(
                mem=mem, query=query
            )
        )
        return result

    @staticmethod
    def to_sql(df, engine, table, if_exists="fail", sep="\t", encoding="utf8"):
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
        from io import StringIO

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

    @staticmethod
    def get_postgres(ip, port):
        # engine = sqlalchemy.create_engine(f'postgresql+psycopg2://postgres:postgres@{ip}:{port}')
        engine = sqlalchemy.create_engine(
            "postgresql+psycopg2://postgres:postgres@{}:{}".format(ip, port)
        )
        return engine

    @staticmethod
    def check_connection(con):
        try:
            con.execute("select 1 as is_alive").fetchall()
            return True
        except Exception:
            raise ValueError("Connection was not established at {}".format(con.url))

    @staticmethod
    def alter_dtype(con, table_name, column_name, dtype):
        return pd.read_sql(
            f"""
    ALTER TABLE {table_name} ALTER COLUMN {column_name} TYPE {dtype}; 
    """,
            con,
        )

    @staticmethod
    def add_column(con, table_name, column_name, column_dtype):
        return pd.read_sql(
            f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_dtype};", con
        )

    @staticmethod
    def col_dtypes(con, table_name):
        return pd.read_sql(
            f"""
    SELECT
            a.attname as "Column",
            pg_catalog.format_type(a.atttypid, a.atttypmod) as "Datatype"
        FROM
            pg_catalog.pg_attribute a
        WHERE
            a.attnum > 0
            AND NOT a.attisdropped
            AND a.attrelid = (
                SELECT c.oid
                FROM pg_catalog.pg_class c
                    LEFT JOIN pg_catalog.pg_namespace n ON n.oid = c.relnamespace
                WHERE c.relname ~ '^({table_name})$'
                    AND pg_catalog.pg_table_is_visible(c.oid)
            );
    """,
            con,
        )

    @staticmethod
    def rename_column(con, table_name, _from, _to):
        con.execute(f'ALTER TABLE {table_name} RENAME COLUMN "{_from}" TO {_to}')

    @staticmethod
    def version(con):
        return con.execute("select version()").fetchone()[0]

    @staticmethod
    def pg_locks_waiting(engine):
        return pd.read_sql(
            """
SELECT count(distinct pid) FROM pg_locks WHERE granted = false;
    """,
            engine,
        )

    @staticmethod
    def pg_dead_tuples_db(engine):
        # -- Total number of dead tuples need to be vacuumed in DB
        return pd.read_sql(
            """
select sum(n_dead_tup) from pg_stat_all_tables;
    """,
            engine,
        )

    @staticmethod
    def pg_dead_tuples_table(engine):
        # -- Total number of dead tuples need to be vacuumed per table

        return pd.read_sql(
            """
select n_dead_tup, schemaname, relname from pg_stat_all_tables;
    """,
            engine,
        )

    @staticmethod
    def pg_last_vaccum(engine):
        # -- Last Vacuum and Analyze time
        return pd.read_sql(
            """
select relname,last_vacuum, last_autovacuum, last_analyze, last_autoanalyze from pg_stat_user_tables;
    """,
            engine,
        )

    @staticmethod
    def pg_user_connections_ratio(engine):
        # -- User Connections Ratio
        return pd.read_sql(
            """
select count(*)*100/(select current_setting('max_connections')::int) from pg_stat_activity;
    """,
            engine,
        )

    @staticmethod
    def pg_check_connections(engine):
        return pd.read_sql(
            """
select client_addr, usename, datname, count(*) from pg_stat_activity group by 1,2,3 order by 4 desc;
    """,
            engine,
        )

    @staticmethod
    def pg_max_transaction_age(engine):
        # -- Long-running transactions are bad because they prevent Postgres from vacuuming old data. This causes database bloat and, in extreme circumstances, shutdown due to transaction ID (xid) wraparound. Transactions should be kept as short as possible, ideally less than a minute.
        # -- Maximum transaction age
        return pd.read_sql(
            """
select client_addr, usename, datname, clock_timestamp() - xact_start as xact_age, clock_timestamp() - query_start as query_age, query from pg_stat_activity order by xact_start, query_start;
    """,
            engine,
        )

    @staticmethod
    def attributes(con):
        return pd.read_sql(
            """
select
*
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'

order by
    t.relname,
    i.relname;

""",
            con,
        )

    @staticmethod
    def pg_table_sizes(engine):
        return pd.read_sql(
            "select relname, pg_size_pretty(pg_total_relation_size(relname::regclass)) as full_size, pg_size_pretty(pg_relation_size(relname::regclass)) as table_size, pg_size_pretty(pg_total_relation_size(relname::regclass) - pg_relation_size(relname::regclass)) as index_size from pg_stat_user_tables order by pg_total_relation_size(relname::regclass) desc;",
            engine,
        )

    @staticmethod
    def pg_unused_indexes(engine):
        # -- idx_scan should not be = 0
        return pd.read_sql(
            """
        select * from pg_stat_all_indexes where idx_scan = 0;
    """,
            engine,
        )

    @staticmethod
    def pg_unused_indexes(engine):
        # -- Write Activity(index usage)
        # -- hot_rate should be close to 100
        return pd.read_sql(
            """
select s.relname, pg_size_pretty(pg_relation_size(relid)), coalesce(n_tup_ins,0) + 2 * coalesce(n_tup_upd,0) - coalesce(n_tup_hot_upd,0) + coalesce(n_tup_del,0) AS total_writes, (coalesce(n_tup_hot_upd,0)::float * 100 / (case when n_tup_upd > 0 then n_tup_upd else 1 end)::float)::numeric(10,2) AS hot_rate, (select v[1] FROM regexp_matches(reloptions::text,E'fillfactor=(d+)') as r(v) limit 1) AS fillfactor from pg_stat_all_tables s join pg_class c ON c.oid=relid order by total_writes desc limit 50;
    """,
            engine,
        )

    @staticmethod
    def pg_db_sizes(engine):
        return pd.read_sql(
            """select datname, pg_size_pretty(pg_database_size(datname)) from pg_database order by pg_database_size(datname);""",
            engine,
        )

    @staticmethod
    def pg_total_table_sizes(engine):
        return pd.read_sql(
            """SELECT nspname || '.' || relname AS "relation", pg_size_pretty(pg_total_relation_size(C.oid)) AS "total_size" FROM pg_class C LEFT JOIN pg_namespace N ON (N.oid = C.relnamespace) WHERE nspname NOT IN ('pg_catalog', 'information_schema') AND C.relkind <> 'i' AND nspname !~ '^pg_toast' ORDER BY pg_total_relation_size(C.oid) DESC;""",
            engine,
        )

    @staticmethod
    def index_info(con):
        return pd.read_sql(
            """
select
    t.relname as table_name,
    i.relname as index_name,
    ix.indisunique as unique_index, 
    array_to_string(array_agg(a.attname), ', ') as column_names
from
    pg_class t,
    pg_class i,
    pg_index ix,
    pg_attribute a
where
    t.oid = ix.indrelid
    and i.oid = ix.indexrelid
    and a.attrelid = t.oid
    and a.attnum = ANY(ix.indkey)
    and t.relkind = 'r'
    and t.relname like 'okta_data'
group by
    t.relname,
    i.relname,
    ix.indisunique
order by
    t.relname,
    i.relname;
""",
            con,
        )


def init_closure_table(con):
    temp_wait = con.execute(
        """
    CREATE TABLE IF NOT EXISTS nodes (
    node_id INT PRIMARY KEY ,
    node_name TEXT NOT NULL,
    role_name TEXT NOT NULL,
    node_histogram TEXT NOT NULL,
    time_bin TEXT NOT NULL
    ) WITHOUT ROWID;"""
    )

    temp_wait = con.execute(
        """
    CREATE TABLE IF NOT EXISTS tree_paths (
    node_ancestor INT NOT NULL,
    node_descendant  INT NOT NULL,
    length INT,
    PRIMARY KEY (node_ancestor, node_descendant),
    FOREIGN KEY (node_ancestor)
        REFERENCES nodes(node_id),
    FOREIGN KEY(node_descendant)
        REFERENCES nodes(node_id)
    );"""
    )

    temp_wait = con.execute(
        """
    CREATE TRIGGER path_len_trigger
    AFTER INSERT ON tree_paths
    FOR EACH ROW
    WHEN NEW.length IS NULL
    BEGIN
      UPDATE tree_paths
      SET length = node_descendant - node_ancestor 
      WHERE rowid = NEW.rowid;
    END;
    """
    )

    return 1


def month_bin_fmt(date):
    from datetime import datetime

    temp = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%fZ")
    temp = datetime.strptime(f"{temp.year}-{temp.month}", "%Y-%m")
    return temp.isoformat(timespec="milliseconds")


def day_bin_fmt(date):
    from datetime import datetime

    temp = datetime.strptime(date, "%Y-%m-%dT%H:%M:%S%fZ")
    temp = datetime.strptime(f"{temp.year}-{temp.month}-{temp.day}", "%Y-%m-%d")
    return temp.isoformat(timespec="milliseconds")


def insert_node(con, node_id, direct_ancestor):
    temp_wait = con.execute(
        f"""
INSERT INTO tree_paths (node_ancestor, node_descendant)
SELECT node_ancestor, {node_id} FROM tree_paths
WHERE node_descendant = {direct_ancestor}
UNION ALL SELECT {node_id}, {node_id}
;"""
    )
    return temp_wait


def query_descendants(con, node_ancestor):
    temp = pd.read_sql_query(
        f"""
select * from nodes n
join tree_paths t
ON (n.node_id = t.node_descendant)
WHERE t.node_ancestor = {node_ancestor}
    ;
""",
        con,
    )
    return temp


def query_ancestors(con, node_descendant):
    temp = pd.read_sql_query(
        f"""
select * from nodes n
join tree_paths t
ON (n.node_id = t.node_ancestor)
WHERE t.node_descendant = {node_descendant}
    ;
""",
        con,
    )
    return temp


def delete_subtree(con, node_ancestor):
    temp_wait = con.execute(
        f"""
DELETE FROM nodes
where node_id IN
(SELECT node_descendant FROM tree_paths
where node_ancestor = {node_ancestor});
"""
    )
    temp_wait = con.execute(
        f"""
DELETE FROM tree_paths
where node_descendant IN
(SELECT node_descendant FROM tree_paths
where node_ancestor = {node_ancestor});
"""
    )
    return 1


def delete_leaf(con, node_descendant):
    temp_wait = con.execute(
        f"""
    DELETE FROM tree_paths
    where node_descendant =  {node_descendant};
"""
    )
    temp_wait = con.execute(
        f"""
    DELETE FROM nodes
    where node_id =  {node_descendant};
"""
    )
    return 1


def serialize_sqlite_buf(con):
    import io

    buff = io.StringIO()
    [buff.write(f"{line}\n") for line in con.iterdump()]
    buff.seek(0)
    return buff


def deserialize_sqlite_buf(con, buf):
    t_str = buf.read()
    con.executescript(t_str)
    return con


def buf_str(buff):
    t_str = buf.read()
    buf.seek(0)
    return t_str


def deserialize_sqlite_bytes(bytes_str):
    t = sqlite3.connect(":memory:")
    t_str = bytes_str.decode(errors="replace")
    t.executescript(t_str)
    t.commit()
    return con


def deserialize_sqlite_bytes(con, bytes_str):
    temp_str = bytes_str.decode(errors="replace")
    con.executescript(temp_str)
    con.commit()
    return con
