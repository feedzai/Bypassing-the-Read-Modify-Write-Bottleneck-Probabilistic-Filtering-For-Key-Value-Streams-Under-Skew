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
                        default=['day'])
    parser.add_argument(
        '--filtering_thresholds',
        type=float,
        nargs='+',
        default=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1],
    )
    parser.add_argument('--entity', type=str, default='merchant_name')
    parser.add_argument(
        '--quantity',
        type=str,
        default='amount_usd',
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
        ## Variance Reduction
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
                event_filtering.TimeDecayedCount('minute'),
                event_filtering.TimeDecayedSum('minute'),
                event_filtering.TimeDecayedAverage('minute'),
                #
                event_filtering.TimeDecayedCount('hour'),
                event_filtering.TimeDecayedSum('hour'),
                event_filtering.TimeDecayedAverage('hour'),
                #
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
            ],
            args.entity,
            args.quantity,
            double_stochastic=True,
            drop_table=False,
            var_alpha=1,
        )

        event_filter.simulate_filtering_double_stochastic(
            half_lifes=args.intensity_function_half_lifes,
            thresholds=args.filtering_thresholds,
            entity=args.entity,
            quantity=args.quantity,
            trials=15,
        )

        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('minute'),
                event_filtering.TimeDecayedSum('minute'),
                event_filtering.TimeDecayedAverage('minute'),
                #
                event_filtering.TimeDecayedCount('hour'),
                event_filtering.TimeDecayedSum('hour'),
                event_filtering.TimeDecayedAverage('hour'),
                #
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
            ],
            args.entity,
            args.quantity,
            double_stochastic=True,
            drop_table=False,
        )

        # Vanilla (full stream filtering)

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
                event_filtering.TimeDecayedCount('hour'),
                event_filtering.TimeDecayedSum('hour'),
                event_filtering.TimeDecayedAverage('hour'),
                #
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
