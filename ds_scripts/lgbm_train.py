"""This script runs hyperparameter optimization for lgbms"""
import os
import argparse

import jinja2
import yaml
import numpy as np

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

# Replaced zaix with native optuna, lightgbm, and sklearn for metrics
import optuna
import lightgbm as lgb
from sklearn.metrics import roc_curve

from smoothoperators import ml
from smoothoperators import feature_profiles
from smoothoperators import event_filtering


def process_data(in_dataframe: pd.DataFrame, label_smoothing: float,
                 label: str):
    df = in_dataframe.copy()
    df[label] = df[label] * (1 - label_smoothing) + label_smoothing / 2
    return df


# --- Custom Metric for Recall at 1% FPR ---
def lgbm_recall_at_fpr_1(preds, train_data):
    """Custom metric for LightGBM to calculate Recall at 1% FPR."""
    y_true = train_data.get_label()
    fpr, tpr, thresholds = roc_curve(y_true, preds)

    # Find the maximum True Positive Rate where False Positive Rate <= 0.01
    valid_fpr_indices = np.where(fpr <= 0.01)[0]
    idx = valid_fpr_indices[-1] if len(valid_fpr_indices) > 0 else 0

    # Return format: (metric_name, eval_result, is_higher_better)
    return 'recall_at_fpr_0_01', tpr[idx], True


def main(
    db_path,
    db_name,
    table_name,
    downsampling_rate,
    temporal_column,
    label_column,
    tables,
    columns,
    join_key,
    parquet_save_path,
    label_smoothing,
    model_config,
    use_previous_downsamples=False,
    split_ratios=None,
    split_timestamps=None,
    join_key_type='UInt64',
    limits=[None, None, None],
):
    if not os.path.exists(parquet_save_path):
        os.makedirs(parquet_save_path, exist_ok=True)

    with feature_profiles.ClickHouseConnector(db_path):
        ds = ml.Downsampler(
            db_name=db_name,
            feature_table_name=table_name,
            downsampling_rate=downsampling_rate,
            temporal_column=temporal_column,
            label_column=label_column,
            split_ratios=split_ratios,
            split_timestamps=split_timestamps,
            join_key_type=join_key_type,
        )

        # --- Fetch and Save Data ---
        df_train = ds.get_examples(
            tables=tables,
            columns=columns,
            join_key=join_key,
            split='train',
            use_previous_downsamples=use_previous_downsamples,
            limit=limits[0],
        )
        print('Fetched training data.')
        train_data = process_data(df_train, label_smoothing, label_column)
        print('Processed data.')
        table = pa.Table.from_pandas(train_data, preserve_index=False)
        pq.write_table(table,
                       os.path.join(parquet_save_path, 'train.parquet'),
                       compression=None)
        print('Saved training data.')

        df_val = ds.get_examples(
            tables=tables,
            columns=columns,
            join_key=join_key,
            split='validation',
            use_previous_downsamples=use_previous_downsamples,
            limit=limits[1],
        )
        print('Fetched validation data.')
        val_data = process_data(df_val, label_smoothing, label_column)
        table = pa.Table.from_pandas(val_data, preserve_index=False)
        pq.write_table(table,
                       os.path.join(parquet_save_path, 'val.parquet'),
                       compression=None)
        print('Saved validation data.')

        df_test = ds.get_examples(
            tables=tables,
            columns=columns,
            join_key=join_key,
            split='test',
            use_previous_downsamples=False,
            limit=limits[2],
        )
        print('Fetched test data')
        # No label smoothing over the test data.
        table = pa.Table.from_pandas(df_test, preserve_index=False)
        pq.write_table(table,
                       os.path.join(parquet_save_path, 'test.parquet'),
                       compression=None)
        print('Saved testing data.')

    # --- Optuna & LightGBM Setup ---
    features = model_config.get('features', [])
    cat_features = model_config.get('categorical_features', [])
    label_col = model_config.get('label_column', 'fraud_label')

    X_train, y_train = train_data[features], train_data[label_col]
    X_val, y_val = val_data[features], val_data[label_col]
    X_test, y_test = df_test[features], df_test[label_col]

    # Create LGBM Datasets directly from memory DataFrames
    lgb_train = lgb.Dataset(X_train,
                            label=y_train,
                            categorical_feature=cat_features,
                            free_raw_data=False)
    lgb_val = lgb.Dataset(X_val,
                          label=y_val,
                          categorical_feature=cat_features,
                          reference=lgb_train,
                          free_raw_data=False)

    optuna_cfg = model_config.get('optuna', {})

    def objective(trial):
        params = model_config.get('architecture_configs', {}).copy()

        stopping_rounds = model_config.get('early_stopping_rounds', 100)

        model = lgb.train(params,
                          lgb_train,
                          num_boost_round=params.get('num_iterations', 500),
                          valid_sets=[lgb_train, lgb_val],
                          valid_names=['train', 'val'],
                          feval=lgbm_recall_at_fpr_1,
                          callbacks=[
                              lgb.early_stopping(
                                  stopping_rounds=stopping_rounds,
                                  first_metric_only=True),
                              lgb.log_evaluation(50)
                          ])

        val_preds = model.predict(X_val, num_iteration=model.best_iteration)
        fpr, tpr, _ = roc_curve(y_val, val_preds)
        valid_fpr_indices = np.where(fpr <= 0.01)[0]
        val_recall = tpr[valid_fpr_indices[-1]] if len(
            valid_fpr_indices) > 0 else 0

        trial.set_user_attr("model", model)

        return val_recall

    # Run Hyperparameter Optimization
    study = optuna.create_study(
        direction=optuna_cfg.get('direction', 'maximize'),
        sampler=optuna.samplers.TPESampler(seed=optuna_cfg.get('seed', 123)))
    study.optimize(objective, n_trials=optuna_cfg.get('n_trials', 1))

    print("\n[Optuna] Best trial:")
    print(f"  Value (Validation Recall @ 1% FPR): {study.best_value}")
    print(f"  Params: {study.best_trial.params}")

    # --- Eval on the Test Set ---
    best_model = study.best_trial.user_attrs["model"]

    # Save the model checkpoint
    exp_folder = model_config.get('experiment_folder', './')
    os.makedirs(os.path.join(exp_folder, 'model_checkpoints'), exist_ok=True)
    best_model_checkpoint_path = os.path.join(exp_folder, 'model_checkpoints',
                                              'model.txt')
    best_model.save_model(best_model_checkpoint_path)

    # Predict and log terminal output
    test_preds = best_model.predict(X_test,
                                    num_iteration=best_model.best_iteration)
    fpr, tpr, _ = roc_curve(y_test, test_preds)

    valid_fpr_indices = np.where(fpr <= 0.01)[0]
    test_recall = tpr[valid_fpr_indices[-1]] if len(
        valid_fpr_indices) > 0 else 0

    print("=" * 40)
    print(f"FINAL TEST METRICS:")
    print(f"Test Recall at 1% FPR: {test_recall:.4f}")
    print("=" * 40)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train lgbm model')
    """
    python lgbm_train.py --db_path /shared/research_projects/smoothOps/data/event_filtering/ibm.chdb --data_config_path ibm_configs/ibm_double_stochastic.yaml --model_config_path ibm_configs/ibm_filtering_zaix.yaml --half_life day --threshold 1e-3 --filter_type double_stochastic
    """
    parser.add_argument('--db_path',
                        type=str,
                        required=True,
                        help='Path to the database')
    parser.add_argument('--data_config_path',
                        type=str,
                        help='Path to yaml containing data configurations')
    parser.add_argument('--model_config_path',
                        type=str,
                        help='Path to the modified model configuration file')
    parser.add_argument(
        '--half_life',
        type=str,
        help='Half live used for the estimation of the intensity functions.')
    parser.add_argument('--threshold',
                        type=float,
                        help='Threshold used for filtering.')
    parser.add_argument('--entity',
                        type=str,
                        help='Group by entity for which to train',
                        required=True)
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--label_smoothing', type=float, default=0)
    parser.add_argument('--filter_type',
                        choices=['vanilla', 'double_stochastic'])
    parser.add_argument('--variance_reduction_alpha', type=float, default=None)
    parser.add_argument('--batch_size', type=int, default=None)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--use_previous_downsamples',
                        action=argparse.BooleanOptionalAction)
    parser.add_argument('--limits',
                        type=int,
                        nargs='+',
                        default=[None, None, None])

    args = parser.parse_args()

    # Note: LightGBM objective name for xentropy is cross_entropy
    if args.label_smoothing == 0:
        objective = 'binary'
    else:
        objective = 'cross_entropy'

    with open(args.data_config_path, 'r') as f:
        config_template = f.read()
        rendered_template = jinja2.Template(config_template).render({
            'trial':
            args.trial,
            'label_smoothing':
            args.label_smoothing,
            'half_life':
            args.half_life,
            'threshold':
            event_filtering.custom_sci(args.threshold)
            if args.threshold is not None else None,
            'entity':
            args.entity,
            'type':
            args.filter_type,
            'var_reduction':
            f'_var_reduction_alpha_{event_filtering.custom_sci(args.variance_reduction_alpha)}'
            if args.variance_reduction_alpha is not None else '',
            'b_size':
            args.batch_size,
            'seed':
            args.seed,
        })
        data_config = yaml.load(rendered_template, Loader=yaml.SafeLoader)

    with open(args.model_config_path, 'r') as f:
        config_template = f.read()
        rendered_template = jinja2.Template(config_template).render({
            'data_path':
            data_config['parquet_save_path'],
            'half_life':
            args.half_life,
            'threshold':
            event_filtering.custom_sci(args.threshold)
            if args.threshold is not None else None,
            'dataset':
            os.path.basename(os.path.dirname(args.db_path)).strip('.chdb'),
            'trial':
            args.trial,
            'objective':
            objective,
            'entity':
            args.entity,
            'type':
            args.filter_type,
            'var_reduction':
            f'_var_reduction_alpha_{event_filtering.custom_sci(args.variance_reduction_alpha)}'
            if args.variance_reduction_alpha is not None else '',
            'b_size':
            args.batch_size,
            'seed':
            args.seed,
        })
        model_config = yaml.load(rendered_template, Loader=yaml.SafeLoader)

    main(db_path=args.db_path,
         **data_config,
         model_config=model_config,
         use_previous_downsamples=args.use_previous_downsamples,
         limits=args.limits + 3 * [None])
