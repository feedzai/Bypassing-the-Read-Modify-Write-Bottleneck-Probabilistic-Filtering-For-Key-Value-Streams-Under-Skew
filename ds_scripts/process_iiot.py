"""Script to process the ibm data."""
import time
import argparse
import yaml
import jinja2

from smoothoperators import feature_profiles


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
def migration_health_check(database_name, table_name):
    return f'''
    SELECT 
        COUNT(*) AS total_records,
        uniqExact(src_ip) AS unique_source_ips,
        SUM(is_tcp) AS total_tcp_packets,
        SUM(is_udp) AS total_udp_packets,
        SUM(fraud_label) AS total_attacks,
        countIf(src_ip = 'MAC_Layer_Traffic') AS non_ip_traffic_count,
        countIf(tcp_dstport = -1) AS missing_tcp_port_count
    FROM {database_name}.{table_name}
    '''


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
    print(
        template_query.render(
            **{
                'db_name': db_name,
                'deduplication_keys': deduplication_keys,
                'columns': columns,
                'file_path': file_path,
                'table_name': table_name,
                'event_time_unix': event_time_unix,
                'partition_by': partition_by,
            }))
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


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline for future plan creation.')

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

    with feature_profiles.ClickHouseConnector(args.session):
        # feature_profiles.drop_table('features', 'events')
        start = time.time()
        results = pipeline.run(
            capture_results=True,
            log_results=True,
        )
        duration = abs(start - time.time())
        for r in results:
            print(r)
        print(duration / 3600)
        print(take_n('features', 'events', 1).iloc[0])
