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
        # Clean data: remove NaNs and Infs
        data_err = df[err_col].replace([np.inf, -np.inf], np.nan).dropna()
        n = len(data_err)

        # 1. Point-wise Average and 95% Confidence Interval
        avg_err = data_err.mean()
        sem = stats.sem(data_err)  # Standard Error of the Mean
        # Using t-distribution for CI (converges to 1.96 for large n)
        margin = sem * stats.t.ppf((1 + confidence) / 2., n - 1)

        # 2. 95th Percentile
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
    parser.add_argument('--half_life', type=str, default='minute')
    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='The chdb path',
    )
    parser.add_argument(
        '--thresholds',
        type=float,
        default=[1, 4, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 8192 * 2])
    parser.add_argument('--volumes',
                        type=float,
                        default=[
                            16e-2, 17e-2, 20e-2, 22e-2, 28e-2, 40e-2, 46e-2,
                            49e-2, 55e-2, 63e-2
                        ])
    parser.add_argument('--entity', type=str, default='dst_ip')
    parser.add_argument('--var_alpha',
                        type=float,
                        default=1.0,
                        help='Alpha in variance reduction.')

    args = parser.parse_args()

    session = args.session
    hf = args.half_life
    window = 'minute'
    n = 500_000
    with feature_profiles.ClickHouseConnector(session):
        for threshold in args.thresholds:
            th = event_filtering.custom_sci(threshold)

            ds_table = f'double_stochastic_time_decayed_sum_{window}_traffic_volume_tcp_{args.entity}_hf_minute_threshold_{th}_0'
            ds_sum = feature_profiles.take_n(
                'features',
                ds_table,
                n,
                order_by='event_time_unix',
            )
            ds_sum.columns = ds_sum.columns.map(lambda x: str(x) + '_ds'
                                                if x != 'join_key' else str(x))

            vr_table = f'double_stochastic_var_reduction_alpha_{event_filtering.custom_sci(args.var_alpha)}_time_decayed_sum_{window}_traffic_volume_tcp_{args.entity}_hf_minute_threshold_{th}_0'
            vr_sum = feature_profiles.take_n('features',
                                             vr_table,
                                             n,
                                             order_by='event_time_unix')
            vr_sum.columns = vr_sum.columns.map(lambda x: str(x) + '_vr'
                                                if x != 'join_key' else str(x))

            vanilla_table = f'time_decayed_sum_{window}_traffic_volume_tcp_{args.entity}_hf_minute_threshold_{th}_0'
            vanilla_sum = feature_profiles.take_n('features',
                                                  vanilla_table,
                                                  n,
                                                  order_by='event_time_unix')
            vanilla_sum.columns = vanilla_sum.columns.map(
                lambda x: str(x) + '_vanilla' if x != 'join_key' else str(x))

            batch_table = 'batch_100_job_aggregations'
            batch_sum = feature_profiles.take_n(
                'features', batch_table, n,
                order_by='event_time_unix').dropna()
            batch_sum.columns = batch_sum.columns.map(
                lambda x: str(x) + '_batch' if x != 'join_key' else str(x))

            true_sum = feature_profiles.take_n(
                'features',
                f'time_decayed_sum_{window}_traffic_volume_tcp_{args.entity}',
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
            merged = pd.merge(merged, true_sum)

            col_true = f'time_decayed_sum_{window}_traffic_volume_tcp_true'
            col_est_ds = f'time_decayed_sum_{window}_traffic_volume_tcp_half_life_{hf}_threshold_{th}_ds'
            col_est_vr = f'time_decayed_sum_{window}_traffic_volume_tcp_half_life_{hf}_threshold_{th}_vr'
            col_est_vanilla = f'time_decayed_sum_{window}_traffic_volume_tcp_half_life_{hf}_threshold_{th}_vanilla'
            col_est_batch = f'time_decayed_sum_{window}_traffic_volume_tcp_batch'
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
            config_list = [('Double Stochastic', 'err_ds'),
                           ('Variance Reduction', 'err_vr'),
                           ('Full Stream', 'err_full'), ('Batch', 'err_batch')]

            thresh_summary = calculate_error_statistics(
                merged, th, config_list)
            print(thresh_summary)
