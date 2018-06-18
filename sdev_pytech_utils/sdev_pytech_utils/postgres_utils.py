"""Python Postgres utility library"""

import sqlalchemy

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


    

