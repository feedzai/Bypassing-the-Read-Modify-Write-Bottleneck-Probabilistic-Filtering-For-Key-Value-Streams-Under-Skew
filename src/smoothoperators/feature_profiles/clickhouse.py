"""Utilities to interact with clickhouse."""
import contextvars
import pandas as pd

import chdb

_CLICKHOUSE_SESSION = contextvars.ContextVar('clickhouse_connection',
                                             default=None)


class ClickHouseConnector:
    """Python context manager for clickhouse sessions. Works by
    setting the clickhouse session global variable.

    Example usage:

    >>> with ClickhouseConnection('db_path.chdb'):
    ...    operations()

    `operations()` will then be performed on the the databses in `db_path`

    """

    def __init__(self, db_path):
        self.db_path = db_path
        self._token = None
        self._session = None

    def __enter__(self):
        self._session = chdb.session.Session(self.db_path)
        self._token = _CLICKHOUSE_SESSION.set(self._session)

        return self

    def __exit__(self, xc_type, exc_value, exc_tb):
        if self._session is not None:
            self._session.close()
        _CLICKHOUSE_SESSION.reset(self._token)
        del self._session
        # self._session = None


def clickhouse_transformation(func, output_format='DataFrame'):
    """Decorator function to then allow sql clickhouse
    transformations. Works by checking the context of the clickhouse
    session and then sending queries to that session.

    Example usage:

    >>> @clickhouse_transformation
    >>> def create_database(arg_1, arg_2): return some_sql_query
    >>> create_database(value_1, value_2)

    This will perform the sql query on whichever clikhouse session is
    open at the moment.

    """

    def wrapper(*args, **kwargs) -> pd.DataFrame:
        session = _CLICKHOUSE_SESSION.get()
        if session is None:
            raise ValueError(
                'Trying to call a transformation without connection to the database'
            )
        query = func(*args, **kwargs)
        return session.query(query, output_format)

    return wrapper
