"""This module offers standard feature transformation functions."""
from typing import Literal, Union, List
import jinja2

from . import clickhouse_transformation, query_templates
from . import clickhouse


@clickhouse_transformation
def pareto_ratio(db_name, table_name, entity):
    return f'''
WITH 
    counts AS (
        SELECT 
            {entity}, 
            count() AS volume
        FROM {db_name}.{table_name}
        GROUP BY {entity}
    ),

    ranked_stats AS (
        SELECT
            volume,
            sum(volume) OVER (ORDER BY volume DESC) AS running_volume,
            sum(volume) OVER () AS total_volume,
            count() OVER () AS total_entities
        FROM counts
    )

SELECT
    count() AS top_entities_count,
    total_entities,
    (count() / total_entities) * 100 AS percentage_of_entities_for_80_pct_volume
FROM ranked_stats
WHERE (running_volume - volume) < (0.8 * total_volume)
GROUP BY total_entities
    '''


@clickhouse.clickhouse_transformation
def add_random_number_column(db_name: str, table_name: str,
                             target_column_name: str):
    """Adds random numbers from the uniform distribution each row."""
    return f'''
ALTER TABLE {db_name}.{table_name}
ADD COLUMN IF NOT EXISTS {target_column_name} Float32
DEFAULT randUniform(0, 1);
    '''


@clickhouse.clickhouse_transformation
def drop_table(db_name: str, table_name: str):
    return f'''DROP TABLE IF EXISTS {db_name}.{table_name}
               SETTINGS max_table_size_to_drop=2000000000000;'''


@clickhouse.clickhouse_transformation
def describe_table(db_name: str, table_name: str):
    return f'DESCRIBE TABLE {db_name}.{table_name};'


@clickhouse.clickhouse_transformation
def get_tables(db_name: str):
    return f'SHOW TABLES FROM {db_name}'


@clickhouse.clickhouse_transformation
def take_n(
    database_name,
    table_name,
    n,
    order_by=None,
):
    if order_by is None:
        return f'SELECT * FROM {database_name}.{table_name} LIMIT {n}'
    return f'SELECT * FROM {database_name}.{table_name} ORDER BY {order_by} LIMIT {n};'


@clickhouse.clickhouse_transformation
def to_day_of_week(
    database_name,
    table_name,
    timestamp_column,
    target_name,
    dtype,
):
    return f'''
    ALTER TABLE {database_name}.{table_name}
    ADD COLUMN IF NOT EXISTS {target_name} {dtype}
    DEFAULT toDayOfWeek({timestamp_column});'''


@clickhouse.clickhouse_transformation
def to_hour(
    database_name,
    table_name,
    timestamp_column,
    target_name,
    dtype,
):
    return f'''
    ALTER TABLE {database_name}.{table_name}
    ADD COLUMN IF NOT EXISTS {target_name} {dtype}
    DEFAULT toHour({timestamp_column});'''


@clickhouse.clickhouse_transformation
def to_day_of_month(
    database_name,
    table_name,
    timestamp_column,
    target_name,
    dtype,
):
    return f'''
    ALTER TABLE {database_name}.{table_name}
    ADD COLUMN IF NOT EXISTS {target_name} {dtype}
    DEFAULT toDayOfMonth({timestamp_column});'''


@clickhouse.clickhouse_transformation
def event_on_weekend(database_name, table_name, timestamp_column, target_name,
                     dtype):
    # TODO (augusto): These transformations can pretty much all be the
    # same function.
    return f'''
    ALTER TABLE {database_name}.{table_name}
    ADD COLUMN IF NOT EXISTS {target_name} {dtype}
    DEFAULT (toDayOfWeek({timestamp_column}) >= 6)'''


@clickhouse.clickhouse_transformation
def estimate_arrival_intensity(
    db_name: str,
    source_table_name: str,
    target_table_name: str,
    partition_by: str,
    order_by: str,
    select_keys: List[str],
    half_lifes: Union[str, float],
    col_names: List[str],
):
    """Computes arrival intensities. Default units for temporal column are miliseconds."""
    window_name = 'w'
    aggregations = [{
        'expression': f'exponentialTimeDecayedCount({half_life})({order_by})',
        'window': window_name,
        'col_name': col_name,
    } for half_life, col_name in zip(half_lifes, col_names)]
    window_expressions = [{
        'name':
        window_name,
        'expression':
        jinja2.Template(query_templates.WINDOW_TEMPLATE).render({
            'partition_by':
            partition_by,
            'order_by':
            order_by,
            'rows': {
                'bottom': 'UNBOUNDED'
            }
        })
    }]
    query = jinja2.Template(
        query_templates.WINDOW_AGGREGATION_TEMPLATE_MILISECONDS).render({
            'db_name':
            db_name,
            'target_table_name':
            target_table_name,
            'order_by': [order_by],
            'select_keys':
            select_keys,
            'aggregations':
            aggregations,
            'source_table_name':
            source_table_name,
            'window_expressions':
            window_expressions,
        })
    print(query)
    return query


@clickhouse.clickhouse_transformation
def aggregation_over_window(
    db_name: str,
    source_table_name: str,
    target_table_name: str,
    partition_by: List[str],
    order_by: List[str],
    order_by_windows: List[str],
    window_type: List[Literal['rows', 'range_']],
    window_bottom: List[Union[str, int]],
    window_top: List[Union[str, int]],
    select_keys: List[str],
    aggregation_expressions: List[str],
    col_names: List[str] = None,
    units: Literal['seconds', 'miliseconds'] = 'seconds',
):
    """Function to compute aggregations over rectangular windows.

    Supports RANGE and ROW limits

    Args:
      db_name: Name of the clickhouse database
      source_table_name: The name of the table from which to fetch the data
        for the aggregation.
      target_table_name: Name of the table that stores the aggregation result
      partition_by: List with columns of the original table to partition by
      order_by: List with columns to order by
      order_by_windows: List with order_by for windows
      window_type: List with either rows or range_. Controls the window type
      window_bottom: List with expressions for the lower end of the window
      window_top: Expression for the upper end of the window
      aggregation_expressions: List with ggregation expresions, e.g,
        sum(column_1)
      col_name: List with names given to the column that stores the
        aggregation results.

    Example usage:

    If we have the table:
       time_stamp  amount  partition_1  partition_2
    0  1546300800     1.0            1            1
    1  1546387200     2.0            1            2
    2  1546473600    -3.0            2            1
    3  1546560000     5.0            2            2
    4  1557874800     0.0            2            1

    And call the function:

    >>> feature_profiles.aggregation_over_window(
           DB_NAME, source_table_name, target_table_name, 'partition_1',
           'time_stamp', 'range_', '10 * day', '0 * day', ['*'],
           'sum(amount)', 'target')

    Yields the table:
       time_stamp  amount  partition_1  partition_2  target
    0  1546300800     1.0            1            1     1.0
    1  1546387200     2.0            1            2     3.0
    2  1546473600    -3.0            2            1    -3.0
    3  1546560000     5.0            2            2     2.0
    4  1557874800     0.0            2            1     0.0


    """
    # TODO (augusto): The arguments this function receives are getting
    # hard to manage. We should refactor it to receive a list of
    # Windows were each element is an element of class Window that has
    # a method `to_clickhouse` that will return an sql expression of
    # the window using clickhouse syntax.
    if units == 'seconds':
        template = query_templates.WINDOW_AGGREGATION_TEMPLATE
    else:
        template = query_templates.WINDOW_AGGREGATION_TEMPLATE_MILISECONDS
    w = lambda wb, wt: {
        'bottom': wb,
        'top': wt
    } if wt is not None else {
        'bottom': wb
    }
    window_expressions = [{
        'expression':
        jinja2.Template(query_templates.WINDOW_TEMPLATE).render({
            'partition_by':
            pb,
            'order_by':
            ob,
            wtype:
            w(wb, wt)
        }),
        'name':
        f'w_{i}'
    } for i, (pb, ob, wtype, wb, wt) in enumerate(
        zip(partition_by, order_by_windows, window_type, window_bottom,
            window_top))]

    aggregations = []
    for w in window_expressions:
        for aggregation_expression in aggregation_expressions:
            aggregations.append({
                'expression':
                aggregation_expression.format(**{'window': w['name']}),
                'window':
                w['name']
            })
    for agg, name in zip(aggregations, col_names):
        agg['col_name'] = name
    query = jinja2.Template(template).render({
        'db_name':
        db_name,
        'target_table_name':
        target_table_name,
        'order_by':
        order_by,
        'select_keys':
        select_keys,
        'aggregations':
        aggregations,
        'source_table_name':
        source_table_name,
        'window_expressions':
        window_expressions,
    })
    print(query)
    return query


@clickhouse.clickhouse_transformation
def sin_features(
    database_name: str,
    table_name: str,
    target_name: str,
    time_column: str,
    period: Union[int, str],
):
    """Applies `sin` to a column of a table. Adds a new column

    Args:
      database_name: Name of the database
      table_name: Name of the table
      target_name: Name of the new column
      time_column: Column to have `sin` applied
      period: Periodicity of the sin function

    """
    template_query = jinja2.Template(query_templates.SIN_TEMPLATE)
    return template_query.render(
        **{
            'database_name': database_name,
            'table_name': table_name,
            'target_name': target_name,
            'time_column': time_column,
            'period': period
        })


@clickhouse.clickhouse_transformation
def cos_features(
    database_name: str,
    table_name: str,
    target_name: str,
    time_column: str,
    period: Union[int, str],
):
    """Applies `cos` to a column of a table. Adds a new column

    Args:
      database_name: Name of the database
      table_name: Name of the table
      target_name: Name of the new column
      time_column: Column to have `cos` applied
      period: Periodicity of the cos function

    """

    template_query = jinja2.Template(query_templates.COS_TEMPLATE)
    return template_query.render(
        **{
            'database_name': database_name,
            'table_name': table_name,
            'target_name': target_name,
            'time_column': time_column,
            'period': period
        })
