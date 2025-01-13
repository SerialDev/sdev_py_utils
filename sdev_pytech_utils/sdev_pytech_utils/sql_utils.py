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


# ------------------------------------------------------------------------- #
#                                   SQLITE                                  #
# ------------------------------------------------------------------------- #

import sqlite3
from pathlib import Path
import pandas as pd


def initialize_sqlite_db(db_path, table_name, col_configs):
    """
    Initialize an SQLite database with the specified configuration.

    :param db_path: Path to the SQLite database file.
    :param table_name: Name of the table to create.
    :param col_configs: Tuple of dictionaries specifying column configurations.
        Each dictionary should have:
            - 'name': Column name
            - 'type': Data type (e.g., TEXT, INTEGER, REAL)
            - 'default': Default value (optional)
    """
    print("\033[36mInitializing SQLite database...\033[0m")

    # Ensure the directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    # Build the CREATE TABLE SQL statement
    cols_definitions = []
    for col in col_configs:
        col_def = f"{col['name']} {col['type']}"
        if "default" in col:
            col_def += f" DEFAULT {repr(col['default'])}"
        cols_definitions.append(col_def)

    create_table_sql = (
        f"CREATE TABLE IF NOT EXISTS {table_name} ("
        + ", ".join(cols_definitions)
        + ");"
    )

    # Connect to the database and execute the statement
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(create_table_sql)
        conn.commit()
        print(f"\033[32mTable '{table_name}' initialized successfully.\033[0m")


def update_sqlite_db(db_path, table_name, col_configs):
    """
    Update an SQLite database table with new columns or create the table if it doesn't exist.

    :param db_path: Path to the SQLite database file.
    :param table_name: Name of the table to update or create.
    :param col_configs: Tuple of dictionaries specifying column configurations.
        Each dictionary should have:
            - 'name': Column name
            - 'type': Data type (e.g., TEXT, INTEGER, REAL)
            - 'default': Default value (optional)
    """
    print("\033[36mUpdating SQLite database...\033[0m")

    # Ensure the directory exists
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()

        # Check if the table exists
        cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}';"
        )
        table_exists = cursor.fetchone() is not None

        if table_exists:
            print(
                f"\033[35mTable '{table_name}' exists. Checking for new columns...\033[0m"
            )
            # Add columns if they don't already exist
            existing_cols = set()
            cursor.execute(f"PRAGMA table_info({table_name});")
            for row in cursor.fetchall():
                existing_cols.add(row[1])  # Column name is in the second field

            for col in col_configs:
                if col["name"] not in existing_cols:
                    col_def = f"{col['name']} {col['type']}"
                    if "default" in col:
                        col_def += f" DEFAULT {repr(col['default'])}"
                    cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {col_def};")
                    print(
                        f"\033[32mAdded column '{col['name']}' to table '{table_name}'.\033[0m"
                    )
        else:
            print(f"\033[35mTable '{table_name}' does not exist. Creating it...\033[0m")
            initialize_sqlite_db(db_path, table_name, col_configs)


def validate_sqlite_with_pandas(db_path, table_name):
    """
    Validate the contents of an SQLite table using pandas.

    :param db_path: Path to the SQLite database file.
    :param table_name: Name of the table to validate.
    :return: Pandas DataFrame containing the table data.
    """
    print("\033[36mValidating SQLite table with pandas...\033[0m")
    with sqlite3.connect(db_path) as conn:
        df = pd.read_sql_query(f"SELECT * FROM {table_name};", conn)
    print("\033[35mTable data:\033[0m")
    print(df)
    return df


def cleanup_sqlite_db(db_path, table_name):
    """
    Clean up a table in the SQLite database by dropping it if it exists.

    :param db_path: Path to the SQLite database file.
    :param table_name: Name of the table to drop.
    """
    print(f"\033[36mCleaning up table '{table_name}'...\033[0m")
    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(f"DROP TABLE IF EXISTS {table_name};")
        conn.commit()
        print(f"\033[32mTable '{table_name}' dropped successfully.\033[0m")


#                                   USAGE                                   #
# ------------------------------------------------------------------------- #

# # Example usage
# initialize_sqlite_db(
#     "../data/db.sqlite",
#     "example_table",
#     (
#         {"name": "id", "type": "INTEGER", "default": None},
#         {"name": "name", "type": "TEXT", "default": ""},
#         {"name": "created_at", "type": "TEXT", "default": "CURRENT_TIMESTAMP"},
#     )
# )

# update_sqlite_db(
#     "../data/db.sqlite",
#     "example_table",
#     (
#         {"name": "updated_at", "type": "TEXT", "default": "CURRENT_TIMESTAMP"},
#         {"name": "description", "type": "TEXT", "default": None},
#     )
# )

# validate_sqlite_with_pandas("../data/db.sqlite", "example_table")
# cleanup_sqlite_db("../data/db.sqlite", "example_table")
# ------------------------------------------------------------------------- #
