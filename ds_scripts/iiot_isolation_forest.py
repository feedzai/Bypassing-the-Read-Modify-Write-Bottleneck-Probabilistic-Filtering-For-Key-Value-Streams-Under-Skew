import argparse

import os
import json
import joblib

import jinja2
import yaml
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, roc_curve)

from smoothoperators import feature_profiles
from smoothoperators import event_filtering


class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)


def recall_at_fpr(scores, labels, target_fpr=0.01):
    fpr, tpr, _ = roc_curve(labels, -1 * scores, pos_label=1)
    closest_fpr_index = np.argmin(np.abs(fpr - target_fpr))
    return tpr[closest_fpr_index]


@feature_profiles.clickhouse_transformation
def fetch_data(db_name, events_table, tables, join_key, columns,
               test_split_timestamp, temporal_column, split):
    join_tables = ''.join([
            f'JOIN {db_name}.{t} ON'\
            f' {db_name}.{events_table}.{join_key} ='
            f' {db_name}.{t}.{join_key}' + '\n'
            for t in tables
        ])
    cols = ', '.join(columns)

    if split == 'train':
        where = f'WHERE {db_name}.{events_table}.{temporal_column} < {test_split_timestamp}'
    else:
        where = f'WHERE {db_name}.{events_table}.{temporal_column} >= {test_split_timestamp}'

    query = f'''
    SELECT {cols}
    FROM {db_name}.{events_table} {join_tables}
    {where}'''
    print(query)
    return query


def train_eval_isolation_forest(train_df,
                                test_df,
                                feature_cols,
                                label_col='fraud_label',
                                forest_args=None):
    """
    Trains an Isolation Forest on a baseline training set and evaluates it on a test set.
    
    Returns:
        dict: A dictionary containing the trained model and evaluation metrics.
    """
    if forest_args is None:
        forest_args = {}

    X_train = train_df[feature_cols]
    X_test = test_df[feature_cols]
    y_test_actual = test_df[label_col]

    iso_forest = IsolationForest(**forest_args)
    iso_forest.fit(X_train)

    raw_scores = iso_forest.decision_function(X_test)
    raw_preds = iso_forest.predict(X_test)
    y_pred = np.where(raw_preds == -1, 1, 0)

    results = {
        'model': iso_forest,
        'recall_at_0.01_fpr': recall_at_fpr(raw_scores, y_test_actual, 0.01),
        'recall_at_0.05_fpr': recall_at_fpr(raw_scores, y_test_actual, 0.05),
        'recall_at_0.1_fpr': recall_at_fpr(raw_scores, y_test_actual, 0.1),
        'recall_at_0.5_fpr': recall_at_fpr(raw_scores, y_test_actual, 0.5),
        'recall_at_1_fpr': recall_at_fpr(raw_scores, y_test_actual, 1),
        'recall_at_5_fpr': recall_at_fpr(raw_scores, y_test_actual, 5),
        'accuracy': accuracy_score(y_test_actual, y_pred),
        'precision': precision_score(y_test_actual, y_pred, zero_division=0),
        'recall': recall_score(y_test_actual, y_pred, zero_division=0),
        'f1_score': f1_score(y_test_actual, y_pred, zero_division=0),
        'confusion_matrix': confusion_matrix(y_test_actual, y_pred),
    }

    return results


def main(chdb_session, db_name, events_table, tables, join_key, columns,
         test_split_timestamp, temporal_column, forest_args, label_column,
         output_folder):
    """The dataset is small enough that we do not worry about downsampling."""
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f'created folder: {output_folder}')

    with feature_profiles.ClickHouseConnector(chdb_session):
        train_df = fetch_data(db_name=db_name,
                              events_table=events_table,
                              tables=tables,
                              join_key=join_key,
                              columns=columns + [label_column],
                              test_split_timestamp=test_split_timestamp,
                              temporal_column=temporal_column,
                              split='train')
        mean = train_df.mean()
        train_df = train_df.fillna(mean)
        test_df = fetch_data(db_name=db_name,
                             events_table=events_table,
                             tables=tables,
                             join_key=join_key,
                             columns=columns + [label_column],
                             test_split_timestamp=test_split_timestamp,
                             temporal_column=temporal_column,
                             split='test')
        test_df = test_df.fillna(mean)

    print(f'Using {len(train_df)} examples for training')
    print(f'Using {len(test_df)} examples for tresting')

    model_and_metrics = train_eval_isolation_forest(
        train_df,
        test_df,
        columns,
        label_column,
        forest_args,
    )
    print(model_and_metrics)

    model_path = os.path.join(output_folder, 'isolation_forest_model.joblib')
    joblib.dump(model_and_metrics['model'], model_path)

    # Save the metrics (excluding the model object)
    json_safe_results = {
        k: v
        for k, v in model_and_metrics.items() if k != 'model'
    }
    metrics_path = os.path.join(output_folder, 'evaluation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(json_safe_results, f, indent=4, cls=NumpyEncoder)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Train lgbm model')

    parser.add_argument(
        '--db_path',
        type=str,
        required=True,
        help='Path to the database',
    )
    parser.add_argument(
        '--config_path',
        type=str,
        help='Path to yaml containing data configurations',
    )
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
                        default='merchant_id')
    parser.add_argument('--volume',
                        type=float,
                        default=None,
                        help='included volume')
    parser.add_argument('--batch',
                        type=int,
                        default=None,
                        help='batch size used for the profiles')
    parser.add_argument('--trial', type=int, default=0)
    parser.add_argument('--filter_type',
                        choices=['vanilla', 'double_stochastic'])
    parser.add_argument('--variance_reduction_alpha', type=float, default=None)

    args = parser.parse_args()

    with open(args.config_path, 'r') as f:
        config_template = f.read()
        rendered_template = jinja2.Template(config_template).render({
            'trial':
            args.trial,
            'half_life':
            args.half_life,
            'threshold':
            event_filtering.custom_sci(args.threshold)
            if args.threshold is not None else None,
            'entity':
            args.entity,
            'var_reduction':
            f'_var_reduction_alpha_{event_filtering.custom_sci(args.variance_reduction_alpha)}'
            if args.variance_reduction_alpha is not None else '',
            'volume':
            f'{event_filtering.custom_sci(args.volume)}'
            if args.volume is not None else '',
            'batch':
            args.batch,
        })
        experiment_config = yaml.load(rendered_template,
                                      Loader=yaml.SafeLoader)
    main(args.db_path, **experiment_config['data'],
         **experiment_config['experiment_config'])
