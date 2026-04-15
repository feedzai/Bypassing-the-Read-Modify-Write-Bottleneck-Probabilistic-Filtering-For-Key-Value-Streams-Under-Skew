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
    parser.add_argument('--entity', type=str, default='editor_id')
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
            1, [82e-2, 87e-2, 92e-2, 95e-2])
        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('day'),
                event_filtering.TimeDecayedSum('day'),
                event_filtering.TimeDecayedAverage('day'),
                #
                event_filtering.TimeDecayedCount('week'),
                event_filtering.TimeDecayedSum('week'),
                event_filtering.TimeDecayedAverage('week'),
                #
                event_filtering.TimeDecayedCount('month'),
                event_filtering.TimeDecayedSum('month'),
                event_filtering.TimeDecayedAverage('month'),
                #
                event_filtering.TimeDecayedCount('4838400000'),
                event_filtering.TimeDecayedSum('4838400000'),
                event_filtering.TimeDecayedAverage('4838400000'),
                #
                event_filtering.TimeDecayedCount('9676800000'),
                event_filtering.TimeDecayedSum('9676800000'),
                event_filtering.TimeDecayedAverage('9676800000'),
                #
                event_filtering.TimeDecayedCount('19353600000'),
                event_filtering.TimeDecayedSum('19353600000'),
                event_filtering.TimeDecayedAverage('19353600000'),
                #
                event_filtering.TimeDecayedCount('38707200000'),
                event_filtering.TimeDecayedSum('38707200000'),
                event_filtering.TimeDecayedAverage('38707200000'),
            ],
            args.entity,
            'edit_size',
        )
