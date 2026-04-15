import argparse

from smoothoperators import feature_profiles
from smoothoperators import event_filtering
import jinja2


@feature_profiles.clickhouse_transformation
def batch_aggregations(aggregation_time_frames):
    window_template = jinja2.Template(feature_profiles.WINDOW_TEMPLATE)
    b_size = 100
    w_expression = window_template.render(
        **{
            'partition_by': 'dst_ip',
            'order_by': 'event_time_unix',
            'rows': {
                'bottom': 'UNBOUNDED',
                'top': str(b_size)
            }
        })
    aggregation_template = jinja2.Template(
        feature_profiles.WINDOW_AGGREGATION_TEMPLATE_MILISECONDS)
    aggregations = []
    for t in aggregation_time_frames:
        count = event_filtering.TimeDecayedCount(t)
        aggregations.append({
            'expression':
            count.unfiltered.format(
                **{
                    'column_name': 'traffic_volume_tcp',
                    'quantity': 'traffic_volume_tcp',
                    'time': 'event_time_unix'
                }),
            'window':
            'w',
            'col_name':
            str(count) + '_traffic_volume_tcp'
        })

        mean = event_filtering.TimeDecayedAverage(t)
        aggregations.append({
            'expression':
            mean.unfiltered.format(
                **{
                    'column_name': 'traffic_volume_tcp',
                    'quantity': 'traffic_volume_tcp',
                    'time': 'event_time_unix'
                }),
            'window':
            'w',
            'col_name':
            str(mean) + '_traffic_volume_tcp'
        })

        sum_ = event_filtering.TimeDecayedSum(t)
        aggregations.append({
            'expression':
            sum_.unfiltered.format(
                **{
                    'column_name': 'traffic_volume_tcp',
                    'quantity': 'traffic_volume_tcp',
                    'time': 'event_time_unix'
                }),
            'window':
            'w',
            'col_name':
            str(sum_) + '_traffic_volume_tcp'
        })
    query = aggregation_template.render(
        **{
            'db_name':
            'features',
            'source_table_name':
            'temp_dst_ip_events',
            'target_table_name':
            f'batch_{b_size}_job_aggregations',
            'order_by': ['event_time_unix', 'dst_ip'],
            'aggregations':
            aggregations,
            'window_expressions': [{
                'expression': w_expression,
                'name': 'w'
            }],
            'select_keys': [
                'join_key',
                'assumeNotNull(dst_ip) AS dst_ip',
                'unix_seconds',
                'event_time_unix',
            ]
        })
    return query


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Pipeline for future plan creation.')

    parser.add_argument(
        '--session',
        type=str,
        required=True,
        help='The chdb path',
    )

    args = parser.parse_args()

    with feature_profiles.ClickHouseConnector(args.session):
        batch_aggregations(['minute', '60000'])
