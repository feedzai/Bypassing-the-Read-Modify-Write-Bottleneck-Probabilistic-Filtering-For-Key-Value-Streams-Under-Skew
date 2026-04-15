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

    parser.add_argument('--intensity_function_half_lifes',
                        type=str,
                        nargs='+',
                        default=['day'])
    parser.add_argument(
        '--filtering_thresholds',
        type=float,
        nargs='+',
        default=[0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.002, 0.003],
    )
    parser.add_argument('--entity', type=str, default='editor_id')
    parser.add_argument(
        '--quantity',
        type=str,
        default='edit_size',
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
        # Double stochastic
        event_filter.simulate_filtering_double_stochastic(
            half_lifes=args.intensity_function_half_lifes,
            thresholds=args.filtering_thresholds,
            entity=args.entity,
            quantity=args.quantity,
            trials=15,
        )

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
                #
            ],
            args.entity,
            args.quantity,
            double_stochastic=True,
            drop_table=False,
        )

        # Variance reduction
        event_filter.simulate_filtering_double_stochastic(
            half_lifes=args.intensity_function_half_lifes,
            thresholds=args.filtering_thresholds,
            entity=args.entity,
            quantity=args.quantity,
            trials=15,
            var_alpha=1,
        )

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
                #
            ],
            args.entity,
            args.quantity,
            double_stochastic=True,
            drop_table=False,
            var_alpha=1,
        )

        # Vanilla filtering
        event_filter.compute_intensity_functions(
            half_lifes=args.intensity_function_half_lifes,
            thresholds=args.filtering_thresholds,
            entity=args.entity,
        )
        event_filter.simulate_filtering(args.entity, 15)
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
                #
            ],
            args.entity,
            args.quantity,
        )
        print('Kept volumes vanilla')
        print(event_filter.kept_volume(args.entity, 0))
        print('Kept volumes double stochastic')
        print(event_filter.kept_volume(args.entity, 0, double_stochastic=True))
        print('Kept volumes variance reduction')
        print(
            event_filter.kept_volume(args.entity,
                                     0,
                                     double_stochastic=True,
                                     var_alpha=1))
