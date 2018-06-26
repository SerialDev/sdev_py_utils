"""Generic SQL related utilities"""


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
