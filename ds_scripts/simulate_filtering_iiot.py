"""This script simulates filtering over our datasets"""
import argparse

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

    parser.add_argument('--intensity_function_half_lifes',
                        type=str,
                        nargs='+',
                        default=['minute'])
    parser.add_argument(
        '--filtering_thresholds',
        type=float,
        nargs='+',
        default=[1, 4, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2],
    )
    parser.add_argument('--entity', type=str, default='dst_ip')
    parser.add_argument(
        '--quantity',
        type=str,
        default='traffic_volume_tcp',
        help='The quantity to use for the temporal aggregations')
    parser.add_argument('--output_folder',
                        type=str,
                        default='filtering_output')

    args = parser.parse_args()

    event_filter = event_filtering.EventFilter(
        'features',
        'events',
        'join_key',
        'unix_seconds',
        'event_time_unix',
    )

    with feature_profiles.ClickHouseConnector(args.session):
        event_filter.simulate_filtering_double_stochastic(
            half_lifes=args.intensity_function_half_lifes,
            thresholds=args.filtering_thresholds,
            entity=args.entity,
            quantity=args.quantity,
            trials=10,
        )

        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('minute'),
                event_filtering.TimeDecayedSum('minute'),
                event_filtering.TimeDecayedAverage('minute'),
                #
                event_filtering.TimeDecayedCount(60000),  # 10 minutes
                event_filtering.TimeDecayedSum(60000),
                event_filtering.TimeDecayedAverage(60000),
                #
            ],
            args.entity,
            args.quantity,
            double_stochastic=True,
            drop_table=False,
        )
        print(event_filter.kept_volume(args.entity, 0, double_stochastic=True))

        event_filter.simulate_filtering_double_stochastic(
            half_lifes=args.intensity_function_half_lifes,
            thresholds=args.filtering_thresholds,
            entity=args.entity,
            quantity=args.quantity,
            trials=10,
            var_alpha=1,
        )

        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('minute'),
                event_filtering.TimeDecayedSum('minute'),
                event_filtering.TimeDecayedAverage('minute'),
                #
                event_filtering.TimeDecayedCount(60000),  # 10 minutes
                event_filtering.TimeDecayedSum(60000),
                event_filtering.TimeDecayedAverage(60000),
                #
            ],
            args.entity,
            args.quantity,
            double_stochastic=True,
            drop_table=False,
            var_alpha=1,
        )
        print(
            event_filter.kept_volume(args.entity,
                                     0,
                                     double_stochastic=True,
                                     var_alpha=1))
        event_filter.compute_intensity_functions(
            half_lifes=args.intensity_function_half_lifes,
            thresholds=args.filtering_thresholds,
            entity=args.entity,
        )
        event_filter.simulate_filtering(args.entity, 15)
        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('minute'),
                event_filtering.TimeDecayedSum('minute'),
                event_filtering.TimeDecayedAverage('minute'),
                #
                event_filtering.TimeDecayedCount(60000),  # 10 minutes
                event_filtering.TimeDecayedSum(60000),
                event_filtering.TimeDecayedAverage(60000),
                #
            ],
            args.entity,
            args.quantity,
        )
        print(event_filter.kept_volume(args.entity, 0))
