import argparse
import pandas as pd
import numpy as np
from scipy import stats
from smoothoperators import feature_profiles
from smoothoperators import event_filtering


def filter_single_entries(
    df,
    entity_col='editor_id',
):
    """
    Removes rows where the entity_col (editor_id) appears only once in the DataFrame.
    """
    counts = df[entity_col].value_counts()
    multi_entry_entities = counts[counts > 1].index
    filtered_df = df[df[entity_col].isin(multi_entry_entities)].copy()
    return filtered_df


def filter_close_to_zero(
    df,
    true_col,
):
    filtered_df = df[df[true_col].abs() > 1]
    return filtered_df


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
    parser.add_argument('--half_life', type=str, default='day')
    parser.add_argument(
        '--thresholds',
        type=float,
        default=[0.001, 0.0011, 0.0012, 0.0013, 0.0014, 0.002, 0.003])
    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='The chdb path',
    )
    parser.add_argument('--volumes',
                        type=float,
                        default=[0.55, 0.88, 0.89, 0.90, 0.91, 0.93, 0.95])
    parser.add_argument('--entity', type=str, default='editor_id')
    parser.add_argument('--var_alpha',
                        type=float,
                        default=1.0,
                        help='Alpha in variance reduction.')

    args = parser.parse_args()

    session = args.session
    hf = args.half_life
    window = '38707200000'
    filter_window = 'day'
    n = 5_000_000
    with feature_profiles.ClickHouseConnector(session):
        for threshold in args.thresholds:
            th = event_filtering.custom_sci(threshold)

            ds_table = f'double_stochastic_time_decayed_sum_{window}_edit_size_{args.entity}_hf_{filter_window}_threshold_{th}_0'
            ds_sum = filter_single_entries(
                feature_profiles.take_n('features', ds_table, n))
            ds_sum.columns = ds_sum.columns.map(lambda x: str(x) + '_ds'
                                                if x != 'join_key' else str(x))

            vr_table = f'double_stochastic_var_reduction_alpha_{event_filtering.custom_sci(args.var_alpha)}_time_decayed_sum_{window}_edit_size_{args.entity}_hf_{filter_window}_threshold_{th}_0'
            vr_sum = filter_single_entries(
                feature_profiles.take_n('features', vr_table, n))
            vr_sum.columns = vr_sum.columns.map(lambda x: str(x) + '_vr'
                                                if x != 'join_key' else str(x))

            vanilla_table = f'time_decayed_sum_{window}_edit_size_{args.entity}_hf_{filter_window}_threshold_{th}_0'
            vanilla_sum = filter_single_entries(
                feature_profiles.take_n('features', vanilla_table, n))
            vanilla_sum.columns = vanilla_sum.columns.map(
                lambda x: str(x) + '_vanilla' if x != 'join_key' else str(x))

            batch_table = 'batch_100_job_aggregations'
            batch_sum = filter_single_entries(
                feature_profiles.take_n('features', batch_table, n))
            batch_sum.columns = batch_sum.columns.map(
                lambda x: str(x) + '_batch' if x != 'join_key' else str(x))

            true_sum = filter_single_entries(
                feature_profiles.take_n(
                    'features',
                    f'time_decayed_sum_{window}_edit_size_{args.entity}', n))
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

            col_true = f'time_decayed_sum_{window}_edit_size_true'
            col_est_ds = f'time_decayed_sum_{window}_edit_size_half_life_{hf}_threshold_{th}_ds'
            col_est_vr = f'time_decayed_sum_{window}_edit_size_half_life_{hf}_threshold_{th}_vr'
            col_est_vanilla = f'time_decayed_sum_{window}_edit_size_half_life_{hf}_threshold_{th}_vanilla'
            col_est_batch = f'time_decayed_sum_{window}_edit_size_batch'
            merged = filter_close_to_zero(merged, col_true)
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
            config_list = [
                ('Double Stochastic', 'err_ds'),
                ('Variance Reduction', 'err_vr'),
                ('Full Stream', 'err_full'),
                #('Batch', 'err_batch')
            ]

            thresh_summary = calculate_error_statistics(
                merged, th, config_list)
            print(thresh_summary)

        # for volume in args.volumes:
        #     v = event_filtering.custom_sci(volume)
        #     fixed_rate_table = f'time_decayed_sum_{window}_edit_size_{args.entity}_at_volume_{v}_0'
        #     fixed_rate_sum = feature_profiles.take_n(
        #         'features',
        #         fixed_rate_table,
        #         n,
        #     )
        #     fixed_rate_sum.columns = fixed_rate_sum.columns.map(lambda x: str(
        #         x) + '_fixed_rate' if x != 'join_key' else str(x))

        #     true_sum = feature_profiles.take_n(
        #         'features',
        #         f'time_decayed_sum_{window}_edit_size_{args.entity}', n)
        #     true_sum.columns = true_sum.columns.map(
        #         lambda x: str(x) + '_true' if x != 'join_key' else str(x))

        #     # Merge sum estimates
        #     merged = pd.merge(
        #         fixed_rate_sum,
        #         true_sum,
        #         on='join_key',
        #     )
        #     col_true = f'time_decayed_sum_{window}_edit_size_true'
        #     col_est_fixed_rate = f'time_decayed_sum_{window}_edit_size_at_volume_{v}_fixed_rate'
        #     epsilon = 1e-3
        #     merged = filter_close_to_zero(merged, col_true)
        #     merged['err_fixed_rate'] = np.abs(
        #         merged[col_true] -
        #         merged[col_est_fixed_rate]) / np.abs(merged[col_true] +
        #                                              epsilon)
        #     config_list = [
        #         ('Fixed rate', 'err_fixed_rate'),
        #     ]

        #     thresh_summary = calculate_error_statistics(merged, v, config_list)
        #     print(thresh_summary)
