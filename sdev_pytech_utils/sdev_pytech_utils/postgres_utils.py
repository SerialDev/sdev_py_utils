"""Python Postgres utility library"""

import sqlalchemy
import pandas as pd


def get_postgres(ip, port):
    # engine = sqlalchemy.create_engine(f'postgresql+psycopg2://postgres:postgres@{ip}:{port}')
    engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:postgres@{}:{}'.format(ip,port ))
    return engine

def check_connection(con):
    try:
        con.execute("select 1 as is_alive").fetchall()
        return True
    except Exception:
        raise ValueError("Connection was not established at {}".format(con.url))

class postgres_utils(object):
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
    def get_engine(ip, port):
        # engine = sqlalchemy.create_engine(f'postgresql+psycopg2://postgres:postgres@{ip}:{port}')
        engine = sqlalchemy.create_engine('postgresql+psycopg2://postgres:postgres@{}:{}'.format(ip,port ))
        return engine

    @staticmethod
    def show_tables(engine):
        return engine.execute("select * from pg_catalog.pg_tables where schemaname != 'information_schema' and schemaname != 'pg_catalog';").fetchall()

    @staticmethod
    def show_running_queries(engine):
        return engine.execute("""SELECT pid, age(query_start, clock_timestamp()), usename, query
FROM pg_stat_activity
WHERE query != '<IDLE>' AND query NOT ILIKE 'pg_stat_activity'
ORDER BY query_start desc;""").fetchall()

    @staticmethod
    def show_running_queries_pd(engine):
        return pd.read_sql("""SELECT pid, age(query_start, clock_timestamp()), usename, query
FROM pg_stat_activity
WHERE query != '<IDLE>' AND query NOT ILIKE 'pg_stat_activity'
ORDER BY query_start desc;""", engine)

    @staticmethod
    def kill_running_query(engine, procpid):
        return engine.execute("""SELECT pg_cancel_backend({procpid})""".format(procpid = procpid))

    @staticmethod
    def kill_idle_query(engine, procpid):
        return engine.execute("""SELECT pg_terminate_backend({procpid})""".format(procpid = procpid))

    @staticmethod
    def show_tables_pd(engine):
        return pd.read_sql("select * from pg_catalog.pg_tables where schemaname != 'information_schema' and schemaname != 'pg_catalog';", engine)

    @staticmethod
    def show_database_sizes(engine):
        return pd.read_sql("SELECT * from pg_user", engine)

    @staticmethod
    def show_database_sizes_pd(engine):
        return pd.read_sql("SELECT * from pg_user", engine)

    @staticmethod
    def show_tables_and_views_usage(engine):
        return engine.execute("""with recursive view_tree(parent_schema, parent_obj, child_schema, child_obj, ind, ord) as (select vtu_parent.view_schema, vtu_parent.view_name, vtu_parent.table_schema, vtu_parent.table_name, '', array[row_number() over (order by view_schema, view_name)]from information_schema.view_table_usage vtu_parent where vtu_parent.view_schema = '<SCHEMA NAME>' and vtu_parent.view_name = '<VIEW NAME>' union all select vtu_child.view_schema, vtu_child.view_name, vtu_child.table_schema, vtu_child.table_name, vtu_parent.ind || '  ', vtu_parent.ord || (row_number() over (order by view_schema, view_name))from view_tree vtu_parent, information_schema.view_table_usage vtu_child where vtu_child.view_schema = vtu_parent.child_schema and vtu_child.view_name = vtu_parent.child_obj) select tree.ind || tree.parent_schema || '.' || tree.parent_obj   || ' depends on ' || tree.child_schema || '.' || tree.child_obj txt, tree.ord from view_tree tree order by ord;""").fetchall()

    @staticmethod
    def show_tables_and_views_usage_pd(engine):
        return pd.read_sql("""with recursive view_tree(parent_schema, parent_obj, child_schema, child_obj, ind, ord) as (select vtu_parent.view_schema, vtu_parent.view_name, vtu_parent.table_schema, vtu_parent.table_name, '', array[row_number() over (order by view_schema, view_name)]from information_schema.view_table_usage vtu_parent where vtu_parent.view_schema = '<SCHEMA NAME>' and vtu_parent.view_name = '<VIEW NAME>' union all select vtu_child.view_schema, vtu_child.view_name, vtu_child.table_schema, vtu_child.table_name, vtu_parent.ind || '  ', vtu_parent.ord || (row_number() over (order by view_schema, view_name))from view_tree vtu_parent, information_schema.view_table_usage vtu_child where vtu_child.view_schema = vtu_parent.child_schema and vtu_child.view_name = vtu_parent.child_obj) select tree.ind || tree.parent_schema || '.' || tree.parent_obj   || ' depends on ' || tree.child_schema || '.' || tree.child_obj txt, tree.ord from view_tree tree order by ord;""", engine)


    @staticmethod
    def show_long_running(engine):
        return engine.execute("""SELECT pid, now() - query_start as "runtime", usename, datname, waiting, state, query FROM  pg_stat_activity WHERE now() - query_start > '2 minutes'::interval and state = 'active' ORDER BY runtime DESC;""").fetchall()

    @staticmethod
    def count_indexes(engine):
        #-- how many indexes are in cache
        return engine.execute("""SELECT sum(idx_blks_read) as idx_read, sum(idx_blks_hit)  as idx_hit, (sum(idx_blks_hit) - sum(idx_blks_read)) / sum(idx_blks_hit) as ratio FROM pg_statio_user_indexes;""").fetchall()

    @staticmethod
    def get_db_encoding(engine, db_name):
        t =  engine.execute("select pg_encoding_to_char(encoding) from pg_database WHERE datname = '{db_name}';".format(db_name = db_name))
        return t.fetchall()

    @staticmethod
    def get_locks_info_pd(engine):
        return pd.read_sql("SELECT * FROM pg_locks pl LEFT JOIN pg_stat_activity psa ON pl.pid = psa.pid;", engine)

    @staticmethod
    def get_db_indexes_pd(engine, db_name):
        return pd.read_sql("""SELECT * FROM pg_indexes WHERE tablename = '{db_name}';""".format(db_name = db_name), engine)

    @staticmethod
    def get_table_sizes(engine):
        df = pd.read_sql("""SELECT
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
        """, engine)
        return df


    @staticmethod
    def get_cache_hit_pd(engine):
        return pd.read_sql( """
SELECT
  sum(heap_blks_read) as heap_read,
  sum(heap_blks_hit)  as heap_hit,
  sum(heap_blks_hit) / (sum(heap_blks_hit) + sum(heap_blks_read)) as ratio
FROM
  pg_statio_user_tables;
""", engine)

    @staticmethod
    def get_index_usage_pd(engine):
        return pd.read_sql( """
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
""", engine)

    @staticmethod
    def get_index_cache_hit_pd(engine):
        return pd.read_sql( """
SELECT
  sum(idx_blks_read) as idx_read,
  sum(idx_blks_hit)  as idx_hit,
  (sum(idx_blks_hit) - sum(idx_blks_read)) / sum(idx_blks_hit) as ratio
FROM
  pg_statio_user_indexes;
""", engine)


    @staticmethod
    def drop_table(engine, table_name):
        engine.execute("DROP TABLE {table_name}".format(table_name = table_name))

    @staticmethod
    def non_seq_query(engine, query, mem='6MB'):
        result = engine.execute("""BEGIN;
SET LOCAL enable_seqscan= off;
SET LOCAL work_mem = '{mem}';
{query}
COMMIT;""".format(mem = mem, query = query))
        return result


    @staticmethod
    def to_sql(df, engine, table, if_exists='fail', sep='\t', encoding='utf8'):
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
        cursor.copy_from(output, table, sep=sep, null='')
        connection.commit()
        cursor.close()
