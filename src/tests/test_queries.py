import tempfile
import pytest
import numpy as np

from smoothoperators import feature_profiles
import jinja2

DB_NAME = 'test_db'


# TODO (augusto): Move general functions to a utils file and separate
# the tests in different files.
@feature_profiles.clickhouse_transformation
def create_database(db_name):
    return f'''
    CREATE DATABASE IF NOT EXISTS {db_name}'''


@feature_profiles.clickhouse_transformation
def select_star(database_name, table_name):
    return f'SELECT * FROM {database_name}.{table_name}'


@feature_profiles.clickhouse_transformation
def create_sin_input(times, table_name, time_column):
    """Creates a table with values to then test the sin transformation."""
    values = ', '.join([f'({t})' for t in times])
    return f'''
    CREATE TABLE IF NOT EXISTS {DB_NAME}.{table_name} (
        {time_column} Float32
    )
    ENGINE = MergeTree
    ORDER BY tuple();

    INSERT INTO {DB_NAME}.{table_name} ({time_column}) VALUES {values}
    '''


@pytest.mark.parametrize('sin_input, period_sql, period_numpy',
                         [(range(5), '2 * PI()', 2 * np.pi),
                          (range(10), 1, 1)])
def test_sin_features(sin_input, period_sql, period_numpy):
    table_name = 'time_table'
    time_column = 'time'
    target_name = 'sin_time'
    expected_output = np.array(
        [np.sin(2 * np.pi * t / period_numpy) for t in sin_input])
    with tempfile.TemporaryDirectory() as tmp_dir:
        with feature_profiles.ClickHouseConnector(tmp_dir):
            create_database(DB_NAME)
            create_sin_input(sin_input, table_name, time_column)
            feature_profiles.sin_features(
                database_name=DB_NAME,
                table_name=table_name,
                target_name=target_name,
                time_column=time_column,
                period=period_sql,
            )
            result = select_star(DB_NAME, table_name)

    assert np.allclose(expected_output, result[target_name].values)


@feature_profiles.clickhouse_transformation
def make_rectangular_windows_input(values, time_stamps, partition_1,
                                   partition_2, table_name):
    """Creates a table with values to then test the rectangular window aggregations."""
    insert_values = ', '.join([
        f'(\'{t}\', {v}, {p1}, {p2})'
        for t, v, p1, p2 in zip(time_stamps, values, partition_1, partition_2)
    ])
    # We order by amount to ensure that even if the input table in not
    # ordered by time stamp the window aggregation will still yield
    # the proper results.
    return f'''
    CREATE TABLE IF NOT EXISTS {DB_NAME}.{table_name} (
        time_stamp DateTime,
        amount Float32,
        partition_1 Int16,
        partition_2 Int16,
    )
    ENGINE = MergeTree
    ORDER BY amount;

    INSERT INTO {DB_NAME}.{table_name} VALUES {insert_values};
    '''


ROW_TEST_INSTANCES = [
    # Test instance 1: No partition_by, over ROWS and using sum
    # aggregation should yield the table:

    #    time_stamp  amount  partition_1  partition_2  target
    # 0  1546300800     1.0            1            1     1.0
    # 1  1546387200     2.0            1            2     3.0
    # 2  1551398400    -3.0            2            1     0.0
    # 3  1554073200     5.0            2            2     5.0
    # 4  1556665200     0.0            2            1     4.0
    (
        [1, 2, -3, 5, 0],
        [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00',
            '2019-03-01 00:00:00', '2019-04-01 00:00:00', '2019-05-01 00:00:00'
        ],
        [1, 1, 2, 2, 2],
        [1, 2, 1, 2, 1],
        ['sum(amount)'],
        [None],
        np.array([1, 3, 0, 5, 4]),
    ),
    # Test instance 2: Partition by partition_1, over ROWS,
    # should yield the table:

    #    time_stamp  amount  partition_1  partition_2  target
    # 0  1546300800     1.0            1            1     1.0
    # 1  1546387200     2.0            1            2     3.0
    # 2  1551398400    -3.0            2            1    -3.0
    # 3  1554073200     5.0            2            2     2.0
    # 4  1556665200     0.0            2            1     2.0
    (
        [1, 2, -3, 5, 0],
        [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00',
            '2019-03-01 00:00:00', '2019-04-01 00:00:00', '2019-05-01 00:00:00'
        ],
        [1, 1, 2, 2, 2],
        [1, 2, 1, 2, 1],
        ['sum(amount)'],
        ['partition_1'],
        np.array([1, 3, -3, 2, 2]),
    ),
    # Test instance 3: Partition by partition_2, over ROWS,
    # should yield the table:

    #     time_stamp  amount  partition_1  partition_2  target
    #  0  1546300800     1.0            1            1     1.0
    #  1  1546387200     2.0            1            2     2.0
    #  2  1551398400    -3.0            2            1    -2.0
    #  3  1554073200     5.0            2            2     7.0
    #  4  1556665200     0.0            2            1    -2.0
    (
        [1, 2, -3, 5, 0],
        [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00',
            '2019-03-01 00:00:00', '2019-04-01 00:00:00', '2019-05-01 00:00:00'
        ],
        [1, 1, 2, 2, 2],
        [1, 2, 1, 2, 1],
        ['sum(amount)'],
        ['partition_2'],
        np.array([1, 2, -2, 7, -2]),
    ),
]


@pytest.mark.parametrize(
    'values, time_stamps, partition_1, partition_2, aggregation_expression, partition_by, expected_values',
    ROW_TEST_INSTANCES)
def test_rectangular_rows(values, time_stamps, partition_1, partition_2,
                          aggregation_expression, partition_by,
                          expected_values):
    source_table_name = 'source_table'
    target_table_name = 'agg'
    target_col_name = 'target'
    with tempfile.TemporaryDirectory() as tmp_dir:
        with feature_profiles.ClickHouseConnector(tmp_dir):
            create_database(DB_NAME)
            make_rectangular_windows_input(values, time_stamps, partition_1,
                                           partition_2, source_table_name)
            feature_profiles.aggregation_over_window(
                DB_NAME, source_table_name, target_table_name, partition_by,
                ['time_stamp'], ['time_stamp'], ['rows'], [3], [0], ['*'],
                aggregation_expression, [target_col_name])
            result = select_star(DB_NAME, target_table_name)
            assert np.allclose(expected_values, result[target_col_name].values)


RANGE_TEST_INSTANCES = [
    # Test instance 1: Partition by None, over RANGE,
    # should yield the table:

    #    time_stamp  amount  partition_1  partition_2  target
    # 0  1546300800     1.0            1            1     1.0
    # 1  1546387200     2.0            1            2     3.0
    # 2  1546473600    -3.0            2            1     0.0
    # 3  1546560000     5.0            2            2     5.0
    # 4  1557874800     0.0            2            1     0.0
    (
        [1, 2, -3, 5, 0],
        [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00',
            '2019-01-03 00:00:00', '2019-01-04 00:00:00', '2019-05-15 00:00:00'
        ],
        [1, 1, 2, 2, 2],
        [1, 2, 1, 2, 1],
        ['sum(amount)'],
        [None],
        np.array([1, 3, 0, 5, 0]),
    ),
    # Test instance 2: Partition by None, over RANGE,
    # should yield the table:

    #    time_stamp  amount  partition_1  partition_2  target
    # 0  1546300800     1.0            1            1     1.0
    # 1  1547251200     2.0            1            2     2.0
    # 2  1548201600    -3.0            2            1    -3.0
    # 3  1549238400     5.0            2            2     5.0
    # 4  1550188800     0.0            2            1     0.0
    (
        [1, 2, -3, 5, 0],
        [
            '2019-01-01 00:00:00', '2019-01-12 00:00:00',
            '2019-01-23 00:00:00', '2019-02-04 00:00:00', '2019-02-15 00:00:00'
        ],
        [1, 1, 2, 2, 2],
        [1, 2, 1, 2, 1],
        ['sum(amount)'],
        [None],
        np.array([1, 2, -3, 5, 0]),
    ),
    # Test instance 3: Partition by None, over RANGE, partition_by
    # partition_1 should yield the table:

    #    time_stamp  amount  partition_1  partition_2  target
    # 0  1546300800     1.0            1            1     1.0
    # 1  1546387200     2.0            1            2     3.0
    # 2  1546473600    -3.0            2            1    -3.0
    # 3  1546560000     5.0            2            2     2.0
    # 4  1557874800     0.0            2            1     0.0
    (
        [1, 2, -3, 5, 0],
        [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00',
            '2019-01-03 00:00:00', '2019-01-04 00:00:00', '2019-05-15 00:00:00'
        ],
        [1, 1, 2, 2, 2],
        [1, 2, 1, 2, 1],
        ['sum(amount)'],
        ['partition_1'],
        np.array([1, 3, -3, 2, 0]),
    ),
]


@pytest.mark.parametrize(
    'values, time_stamps, partition_1, partition_2, aggregation_expression, partition_by, expected_values',
    RANGE_TEST_INSTANCES)
def test_rectangular_range(values, time_stamps, partition_1, partition_2,
                           aggregation_expression, partition_by,
                           expected_values):
    source_table_name = 'source_table'
    target_table_name = 'agg'
    target_col_name = 'target'
    with tempfile.TemporaryDirectory() as tmp_dir:
        with feature_profiles.ClickHouseConnector(tmp_dir):
            create_database(DB_NAME)
            make_rectangular_windows_input(values, time_stamps, partition_1,
                                           partition_2, source_table_name)
            feature_profiles.aggregation_over_window(
                DB_NAME, source_table_name, target_table_name, partition_by,
                ['time_stamp'], ['time_stamp'], ['range_'], ['10 * day'],
                ['0 * day'], ['*'], aggregation_expression, [target_col_name])
            result = select_star(DB_NAME, target_table_name)
            assert np.allclose(expected_values, result[target_col_name].values)


DECAYED_COUNT_TEST_INSTANCES = [
    #    time_stamp  partition     target
    # 0  1546300800          1        1.0
    # 1  1546387200          1        1 + 1/e
    # 2  1546473600          1        1 + 1/e + 1/e**2
    # 3  1546560000          1        1 + 1/e + 1/e**2 + 1/e**3
    # 4  1557874800          1        1
    (
        [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00',
            '2019-01-03 00:00:00', '2019-01-04 00:00:00', '2019-05-15 00:00:00'
        ],
        [1, 1, 1, 1, 1],
        'partition',
        np.array([
            1,
            1 + np.sum([np.exp(-i) for i in range(1, 2)]),
            1 + np.sum([np.exp(-i) for i in range(1, 3)]),
            1 + np.sum([np.exp(-i) for i in range(1, 4)]),
            1,
        ]),
    ),

    #    time_stamp  partition     target
    # 0  1546300800          1        1.0
    # 1  1546387200          2        1.0
    # 2  1546473600          1        1 + 1/e**2
    # 3  1546560000          2        1 + 1/e**2
    # 4  1557874800          1        1 + 1/e**2 + 1/e**4
    (
        [
            '2019-01-01 00:00:00', '2019-01-02 00:00:00',
            '2019-01-03 00:00:00', '2019-01-04 00:00:00', '2019-01-05 00:00:00'
        ],
        [1, 2, 1, 2, 1],
        'partition',
        np.array([
            1,
            1,
            1 + np.sum([np.exp(-i) for i in range(2, 3, 2)]),
            1 + np.sum([np.exp(-i) for i in range(2, 3, 2)]),
            1 + np.sum([np.exp(-i) for i in range(2, 5, 2)]),
        ]),
    ),
]


@feature_profiles.clickhouse_transformation
def make_moving_sum_input(time_stamps, partition, table_name):
    """Creates a table with values to then test the rectangular window aggregations."""
    insert_values = ', '.join([
        f'(toUnixTimestamp(\'{t}\') * 1000, {p})'  # Use miliseconds for window aggregations.
        for t, p in zip(time_stamps, partition)
    ])
    return f'''
    CREATE TABLE IF NOT EXISTS {DB_NAME}.{table_name} (
        time_stamp UInt64,
        partition Int16,
    )
    ENGINE = MergeTree
    ORDER BY time_stamp;

    INSERT INTO {DB_NAME}.{table_name} VALUES {insert_values};
    '''


@pytest.mark.parametrize(
    'time_stamps, partition, partition_by, expected_values',
    DECAYED_COUNT_TEST_INSTANCES)
def test_exponential_moving_sum(time_stamps, partition, partition_by,
                                expected_values):
    source_table_name = 'source_table'
    target_table_name = 'moving_sum'
    target_col_name = 'moving_sum'
    with tempfile.TemporaryDirectory() as tmp_dir:
        with feature_profiles.ClickHouseConnector(tmp_dir):
            create_database(DB_NAME)
            make_moving_sum_input(time_stamps, partition, source_table_name)
            feature_profiles.estimate_arrival_intensity(
                DB_NAME, source_table_name, target_table_name, partition_by,
                'time_stamp', ['*'], ['day'], [target_col_name])
            result = select_star(DB_NAME, target_table_name)
            print(result)
            assert np.allclose(expected_values, result[target_col_name].values)
