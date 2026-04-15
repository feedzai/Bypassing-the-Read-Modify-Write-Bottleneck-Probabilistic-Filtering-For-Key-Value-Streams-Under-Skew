import argparse
import pandas as pd
import numpy as np
from scipy import stats
from smoothoperators import feature_profiles
from smoothoperators import event_filtering


def calculate_error_statistics(df, threshold_str, configs):
    """
    Computes Point-wise Avg (with 95% CI), P95, and Full Stream Error.
    """
    stats_list = []
    confidence = 0.95

    for name, err_col in configs:
        data_err = df[err_col].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(data_err)

        avg_err = data_err.mean()
        sem = stats.sem(data_err)  # Standard Error of the Mean
        margin = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        p95_err = data_err.quantile(0.95)

        stats_list.append({
            'Threshold': threshold_str,
            'Configuration': name,
            'Avg. Rel. Error (95% CI)': f"${avg_err:.4f} \\pm {margin:.4f}$",
            '95th Percentile': f"{p95_err:.4f}",
        })

    return pd.DataFrame(stats_list)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Approximation errors over dataset.')
    parser.add_argument('--half_life', type=str, default='day')
    parser.add_argument('--thresholds',
                        type=float,
                        default=[1e-3, 5e-3, 1e-2, 5e-2, 1e-1, 1])
    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='The chdb path',
    )
    parser.add_argument('--volumes',
                        type=float,
                        default=[33e-2, 45e-2, 54e-2, 78e-2, 90e-2, 95e-2])
    parser.add_argument('--entity', type=str, default='merchant_name')
    parser.add_argument('--var_alpha',
                        type=float,
                        default=1.0,
                        help='Alpha in variance reduction.')

    args = parser.parse_args()

    session = args.session
    hf = args.half_life
    window = 'month'
    n = 50_000
    with feature_profiles.ClickHouseConnector(session):
        for threshold in args.thresholds:
            th = event_filtering.custom_sci(threshold)

            ds_table = f'double_stochastic_time_decayed_sum_{window}_amount_usd_{args.entity}_hf_{hf}_threshold_{th}_0'
            ds_sum = feature_profiles.take_n('features',
                                             ds_table,
                                             n,
                                             order_by='event_time_unix')
            ds_sum.columns = ds_sum.columns.map(lambda x: str(x) + '_ds'
                                                if x != 'join_key' else str(x))

            vr_table = f'double_stochastic_var_reduction_alpha_{event_filtering.custom_sci(args.var_alpha)}_time_decayed_sum_{window}_amount_usd_{args.entity}_hf_{hf}_threshold_{th}_0'
            vr_sum = feature_profiles.take_n('features',
                                             vr_table,
                                             n,
                                             order_by='event_time_unix')
            vr_sum.columns = vr_sum.columns.map(lambda x: str(x) + '_vr'
                                                if x != 'join_key' else str(x))

            vanilla_table = f'time_decayed_sum_{window}_amount_usd_{args.entity}_hf_{hf}_threshold_{th}_0'
            vanilla_sum = feature_profiles.take_n('features',
                                                  vanilla_table,
                                                  n,
                                                  order_by='event_time_unix')
            vanilla_sum.columns = vanilla_sum.columns.map(
                lambda x: str(x) + '_vanilla' if x != 'join_key' else str(x))

            batch_table = 'batch_100_job_aggregations'
            batch_sum = feature_profiles.take_n('features',
                                                batch_table,
                                                n,
                                                order_by='event_time_unix')
            batch_sum.columns = batch_sum.columns.map(
                lambda x: str(x) + '_batch' if x != 'join_key' else str(x))

            true_sum = feature_profiles.take_n(
                'features',
                f'time_decayed_sum_{window}_amount_usd_{args.entity}',
                n,
                order_by='event_time_unix')
            true_sum.columns = true_sum.columns.map(
                lambda x: str(x) + '_true' if x != 'join_key' else str(x))

            # Merge sum estimates
            merged = pd.merge(
                vr_sum,
                ds_sum,
                on='join_key',
            )
            merged_ = pd.merge(
                batch_sum,
                vanilla_sum,
                on='join_key',
            )
            merged = pd.merge(merged, merged_, on='join_key')

            # Merge ground truth
            merged = pd.merge(merged, true_sum)

            col_true = f'time_decayed_sum_{window}_amount_usd_true'
            col_est_ds = f'time_decayed_sum_{window}_amount_usd_half_life_{hf}_threshold_{th}_ds'
            col_est_vr = f'time_decayed_sum_{window}_amount_usd_half_life_{hf}_threshold_{th}_vr'
            col_est_vanilla = f'time_decayed_sum_{window}_amount_usd_half_life_{hf}_threshold_{th}_vanilla'
            col_est_batch = f'time_decayed_sum_{window}_amount_usd_batch'
            epsilon = 1e-3
            merged['err_ds'] = np.abs(merged[col_true] - merged[col_est_ds]
                                      ) / np.abs(merged[col_true] + epsilon)
            merged['err_vr'] = np.abs(merged[col_true] - merged[col_est_vr]
                                      ) / np.abs(merged[col_true] + epsilon)
            merged['err_full'] = np.abs(merged[col_true] -
                                        merged[col_est_vanilla]) / np.abs(
                                            merged[col_true] + epsilon)
            merged['err_batch'] = np.abs(merged[col_true] -
                                         merged[col_est_batch]) / np.abs(
                                             merged[col_true] + epsilon)

            # 5. Generate Stats for this threshold
            config_list = [('Variance reduction', 'err_ds'),
                           ('Double stochastic', 'err_vr'),
                           ('Full Stream', 'err_full'), ('Batch', 'err_batch')]

            thresh_summary = calculate_error_statistics(
                merged, th, config_list)
            print(thresh_summary)

        for volume in args.volumes:
            v = event_filtering.custom_sci(volume)
            fixed_rate_table = f'time_decayed_sum_{window}_amount_usd_{args.entity}_at_volume_{v}_0'
            fixed_rate_sum = feature_profiles.take_n(
                'features',
                fixed_rate_table,
                n,
            )
            fixed_rate_sum.columns = fixed_rate_sum.columns.map(lambda x: str(
                x) + '_fixed_rate' if x != 'join_key' else str(x))

            true_sum = feature_profiles.take_n(
                'features',
                f'time_decayed_sum_{window}_amount_usd_{args.entity}', n)
            true_sum.columns = true_sum.columns.map(
                lambda x: str(x) + '_true' if x != 'join_key' else str(x))

            # Merge sum estimates
            merged = pd.merge(
                fixed_rate_sum,
                true_sum,
                on='join_key',
            )
            col_true = f'time_decayed_sum_{window}_amount_usd_true'
            col_est_fixed_rate = f'time_decayed_sum_{window}_amount_usd_at_volume_{v}_fixed_rate'
            epsilon = 1e-3
            merged['err_fixed_rate'] = np.abs(
                merged[col_true] -
                merged[col_est_fixed_rate]) / np.abs(merged[col_true] +
                                                     epsilon)
            config_list = [
                ('Fixed rate', 'err_fixed_rate'),
            ]

            thresh_summary = calculate_error_statistics(merged, v, config_list)
            print(thresh_summary)
