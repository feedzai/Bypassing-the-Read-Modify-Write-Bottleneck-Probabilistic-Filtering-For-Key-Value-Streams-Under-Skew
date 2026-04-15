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
    parser.add_argument('--entity', type=str, default='merchant_name')
    args = parser.parse_args()

    event_filter = event_filtering.RandomFilter(
        'features',
        'events',
        'join_key',
        'unix_seconds',
        'event_time_unix',
    )

    with feature_profiles.ClickHouseConnector(args.session):
        event_filter.simulate_filtering_at_volumes(
            1, [33e-2, 45e-2, 54e-2, 78e-2, 90e-2, 95e-2])
        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('minute'),
                event_filtering.TimeDecayedSum('minute'),
                event_filtering.TimeDecayedAverage('minute'),
                event_filtering.TimeDecayedCount('hour'),
                event_filtering.TimeDecayedSum('hour'),
                event_filtering.TimeDecayedAverage('hour'),
                event_filtering.TimeDecayedCount('day'),
                event_filtering.TimeDecayedSum('day'),
                event_filtering.TimeDecayedAverage('day'),
                event_filtering.TimeDecayedCount('week'),
                event_filtering.TimeDecayedSum('week'),
                event_filtering.TimeDecayedAverage('week'),
                event_filtering.TimeDecayedCount('month'),
                event_filtering.TimeDecayedSum('month'),
                event_filtering.TimeDecayedAverage('month'),
            ],
            args.entity,
            'amount_usd',
        )
