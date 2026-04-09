# Interacting with clickhouse

We interact with clickhouse using the `chdb` python wrapper with a bit of our
code in the mix in order to seamlessly write complex `SQL` pipelines. Most of
this code exists under the `feature_profiles/clickhouse.py` file that contains
the necessary tools to interact with clickhouse. There are main two components:

1. A decorator `@clickhouse_transformation` that can be placed in any function
   that returns an SQL query as a python string.
2. A context manager `ClickhouseConnector` that handles the session to be used
   by the transformations.
   
Example usage:

```python
@feature_profiles.clickhouse_transformation
def take_n(
    database_name,
    table_name,
    n,
):
    return f'SELECT * FROM {database_name}.{table_name} LIMIT {n}'
with feature_profiles.ClickHouseConnector(args.session):
    data_frame = take('database_name', 'table_name', 5)
```

In the previous code snippet `data_frame` will be a dataframe with the result of
taking the first 5 examples from the `'table_name'` table.

The `query_templates.py` templates in `jinja2` syntax for aggregations and
other transformations that are commonly used across the project.

Finally, the `pipeline.py` file contains the definition of the `Pipeline` class
object. This is a simple class that allows us to add transformations to it and
then execute them:

```python
pipeline = Pipeline()
pipeline.add_transformation(
    take_n,
	{'database_name': 'db_name',
	 'table_name'; 'some_table',
	 'n' = 5},
	 name='Take 5 from some_table'
)
with feature_profiles.ClickHouseConnector(args.session):
    pipeline.run(
        capture_results=True,
        log_results=True,
    )
```


## Machine learning jobs

The code for machine learning jobs is stored under the `ml` folder. Here we have
the `downsampler.py` and `metrics` files that contain the `Downsampler` class
and some metrics respectively.

The `Downsampler` class is responsible from downsampling examples for training,
validation and testing from previously processed data-sets. Example usage:

```python
with feature_profiles.ClickHouseConnector(path_to_db):
    ds = ml.Downsampler(db_name='features',
                        feature_table_name='events',
                        downsampling_rate=10,
                        temporal_column='event_time_unix',
                        label_column='fraud_label'
                        split_timestamps=[
                          [1651363200000, 1669852800000]
                          [1669852800000, 1672531200000]
                          [1675209600000, 1680307200000]])
    df = ds.get_examples(['aggregations_payment_card_id',
                          'aggregations_merchant_id'], ['*'],
                          'join_key', split='validation')
```

This will return a dataframe with the events to be used for validation.
