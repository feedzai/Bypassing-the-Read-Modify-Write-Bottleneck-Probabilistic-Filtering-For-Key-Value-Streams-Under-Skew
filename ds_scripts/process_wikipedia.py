import time
import argparse
import yaml
import jinja2

from smoothoperators import feature_profiles
from smoothoperators import event_filtering


@feature_profiles.clickhouse_transformation
def create_database(db_name):
    return f'''
    CREATE DATABASE IF NOT EXISTS {db_name}'''


@feature_profiles.clickhouse_transformation
def take_n(
    database_name,
    table_name,
    n,
):
    return f'SELECT * FROM {database_name}.{table_name} LIMIT {n}'


@feature_profiles.clickhouse_transformation
def migrate_from_parquet(
    db_name,
    table_name,
    deduplication_keys,
    columns,
    file_path,
    event_time_unix,
    partition_by,
):
    """Handles the migration of the parquet files into a clickhouse db

    Args:
      db_name: Name of the database.
      table_name: Name of the table to which the data is imported.
      deduplication_keys: List with the keys for the SELECT distinct.
      colums: List of Dict with keys 'source_name' and 'target_name'. This is
        rendered as 'SELECT source_name AS target_name'.
      file_path: Path to the folder with the parquet files.
      event_time_uninx: Name of the column that contains the event timestamps
        in the original .parquet files.

    """
    template_query = jinja2.Template(
        feature_profiles.PARQUET_MIGRATION_TEMPLATE)
    return template_query.render(
        **{
            'db_name': db_name,
            'deduplication_keys': deduplication_keys,
            'columns': columns,
            'file_path': file_path,
            'table_name': table_name,
            'event_time_unix': event_time_unix,
            'partition_by': partition_by,
        })


@feature_profiles.clickhouse_transformation
def find_split_labels(
    db_name,
    table_name,
):
    return f'''
    SELECT
        quantile(0.80)(event_timestamp_UTC) AS train_cutoff,
        quantile(0.90)(event_timestamp_UTC) AS val_cutoff,
        count() AS total_vandalism_events
    FROM {db_name}.{table_name}
    WHERE fraud_label = 1;
    '''


@feature_profiles.clickhouse_transformation
def min_max_times(
    db_name,
    table_name,
):
    return f'''
    SELECT
        min(event_timestamp_UTC) AS min_time,
        max(event_timestamp_UTC) AS max_time
    FROM {db_name}.{table_name}
    '''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline to migrate the wikipedia data')

    parser.add_argument(
        '--migration_config_args',
        type=str,
        required=True,
        help=
        'Path to the yaml containing the arguments for the database migration.'
    )

    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='The chdb path',
    )

    args = parser.parse_args()

    pipeline = feature_profiles.Pipeline()

    with open(args.migration_config_args, 'r', encoding='utf-8') as f:
        migration_args = yaml.load(f, Loader=yaml.SafeLoader)
    db_name = migration_args['db_name']

    pipeline.add_transformation(create_database, {'db_name': db_name})

    pipeline.add_transformation(migrate_from_parquet,
                                migration_args,
                                name='migration_from_parquet')

    pipeline.add_transformation(
        feature_profiles.to_day_of_week, {
            'database_name': db_name,
            'table_name': 'events',
            'timestamp_column': 'event_timestamp_UTC',
            'target_name': 'day_of_week',
            'dtype': 'UInt8',
        })

    pipeline.add_transformation(
        feature_profiles.to_hour, {
            'database_name': db_name,
            'table_name': 'events',
            'timestamp_column': 'event_timestamp_UTC',
            'target_name': 'hour_of_day',
            'dtype': 'UInt16',
        })

    pipeline.add_transformation(
        feature_profiles.event_on_weekend, {
            'database_name': db_name,
            'table_name': 'events',
            'timestamp_column': 'event_timestamp_UTC',
            'target_name': 'weekend',
            'dtype': 'Bool',
        })

    pipeline.add_transformation(
        feature_profiles.sin_features, {
            'database_name': db_name,
            'table_name': 'events',
            'target_name': 'sin_day_of_week',
            'time_column': 'day_of_week',
            'period': 7,
        })

    pipeline.add_transformation(
        feature_profiles.cos_features, {
            'database_name': db_name,
            'table_name': 'events',
            'target_name': 'cos_day_of_week',
            'time_column': 'day_of_week',
            'period': 7,
        })

    pipeline.add_transformation(
        feature_profiles.sin_features, {
            'database_name': db_name,
            'table_name': 'events',
            'target_name': 'sin_hour_of_day',
            'time_column': 'hour_of_day',
            'period': 24,
        })

    pipeline.add_transformation(
        feature_profiles.cos_features, {
            'database_name': db_name,
            'table_name': 'events',
            'target_name': 'cos_hour_of_day',
            'time_column': 'hour_of_day',
            'period': 24,
        })

    pipeline.add_transformation(
        take_n,
        {
            'database_name': db_name,
            'table_name': 'events',
            'n': 20000
        },
        name='Take from events table',
    )
    with feature_profiles.ClickHouseConnector(args.session):
        start = time.time()
        results = pipeline.run(
            capture_results=True,
            log_results=True,
        )
        duration = abs(start - time.time())
        for r in results:
            print(r)
        print(duration / 3600)
        print(find_split_labels('features', 'events'))
        print(min_max_times('features', 'events'))
