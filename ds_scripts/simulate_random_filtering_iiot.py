import argparse
import math

from smoothoperators import feature_profiles
from smoothoperators import event_filtering


@feature_profiles.clickhouse_transformation
def take_n(
    database_name,
    table_name,
    columns,
    n,
):
    """Fetches rows from a table"""
    cols = ','.join(columns)
    return f'SELECT {cols} FROM {database_name}.{table_name} ORDER BY event_time_unix LIMIT {n}'


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Script to simulate filtering over datasets.')
    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='The chdb path',
    )
    parser.add_argument('--entity', type=str, default='dst_ip')
    args = parser.parse_args()

    event_filter = event_filtering.RandomFilter(
        'features',
        'events',
        'join_key',
        'unix_seconds',
        'event_time_unix',
    )

    with feature_profiles.ClickHouseConnector(args.session):
        event_filter.simulate_filtering_at_volumes(5, [
            16e-2, 17e-2, 20e-2, 22e-2, 28e-2, 40e-2, 46e-2, 49e-2, 55e-2,
            63e-2
        ])
        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('minute'),
                event_filtering.TimeDecayedSum('minute'),
                event_filtering.TimeDecayedAverage('minute'),
                #
                event_filtering.TimeDecayedCount(60000),
                event_filtering.TimeDecayedSum(60000),
                event_filtering.TimeDecayedAverage(60000),
                #
            ],
            args.entity,
            'traffic_volume_tcp',
        )
