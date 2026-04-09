"""Contains a clickhouse downsampler."""
from typing import List, Literal, Optional

from ..feature_profiles import clickhouse_transformation


class Downsampler:
    """Downsampler class for model training. Allows us to connect to a
    database and extract examples for training and validation.

    Example usage:

    >>> with feature_profiles.ClickHouseConnector(path_to_db):
    >>>     ds = ml.Downsampler(db_name='features',
                                feature_table_name='events',
                                downsampling_rate=10,
                                temporal_column='event_time_unix',
                                label_column='fraud_label'
                                train_validation_ratio=.8)
    >>>     df = ds.get_examples(['aggregations_payment_card_id',
                                  'aggregations_merchant_id'], ['*'],
                                  'join_key', split='validation')

    Or:

    >>> with feature_profiles.ClickHouseConnector(path_to_db):
    >>>     ds = ml.Downsampler(db_name='features',
                                feature_table_name='events',
                                downsampling_rate=10,
                                temporal_column='event_time_unix',
                                label_column='fraud_label'
                                split_timestamps=[
                                  [1651363200000, 1669852800000]
                                  [1669852800000, 1672531200000]
                                  [1675209600000, 1680307200000]])


    It is assumed that the database will have a table with main
    features, e.g, `fetaure_table_name`, and that this table, as well
    as the tables containing other features such as aggregations, have
    a `join_key` columns that will allow joining the tables.

    """

    def __init__(self,
                 db_name: str,
                 feature_table_name: str,
                 downsampling_rate: float,
                 temporal_column: str,
                 label_column: str,
                 split_ratios: Optional[List[float]] = None,
                 split_timestamps: Optional[List[List[int]]] = None,
                 join_key_type: Optional[str] = 'UInt64'):
        """

        Args:
          db_name: Name of the database to use
          features_table_name: Name of the main table to use
          downsampling_rate: Number of non-fraud to frau examples
          temporal_column: Column with the timestamps. We use this column to
            split the data ensuring that the examples in validation are more
            recent that th examples in training
          label_column: Name of the column with the ground truths for training
          split_rations: Percentage of examples for training, validation and testing.
          split_timestapms: Specific timestamps for splitting data into
            training/validation/testing.

        """
        if split_ratios is not None and split_timestamps is not None:
            raise ValueError(
                'Cannot provide split ratios and time stamps at the same time.'
            )

        # Default 80% for training, then 50% of the second split for
        # valition and the rest for training.
        if split_ratios is None and split_timestamps is None:
            split_ratios = [0.8, 0.5]
        self.db_name = db_name
        self.feature_table_name = feature_table_name
        self.downsampling_rate = downsampling_rate
        self.temporal_column = temporal_column
        self.label_column = label_column
        self.join_key_type = join_key_type

        if split_ratios:
            print('Computing split time stamp')
            self.split_timestamp_train_bottom = None
            self.split_timestamp_train_top = self._compute_split_timestamp(
                split_ratios[0]).iloc[0, 0]
            self.split_timestamp_val_bottom = self.split_timestamp_train_top
            self.split_timestamp_val_top = self._compute_split_timestamp(
                split_ratios[1], self.split_timestamp_val_bottom).iloc[0, 0]
            self.split_timestamp_test_bottom = self.split_timestamp_val_top
            self.split_timestamp_test_top = None
        if split_timestamps:
            (self.split_timestamp_train_bottom,
             self.split_timestamp_train_top) = split_timestamps[0]
            (self.split_timestamp_val_bottom,
             self.split_timestamp_val_top) = split_timestamps[1]
            (self.split_timestamp_test_bottom,
             self.split_timestamp_test_top) = split_timestamps[2]

        print('Computing downsampling ratio')
        self.ds_ratio_train = self._compute_selection_ratio('train').iloc[0, 0]
        self.ds_ratio_val = self._compute_selection_ratio('validation').iloc[0,
                                                                             0]

        print('Downsampling ratio train:')
        print(self.ds_ratio_train)
        print('Downsampling ratio validation:')
        print(self.ds_ratio_val)

        print('Available tables in the database')
        print(self.show_tables())

        print('Dataset stats')
        print(self._stats('train'))
        print(self._stats('validation'))
        print(self._stats('test'))

    @clickhouse_transformation
    def _compute_split_timestamp(
        self,
        quantile: float,
        min_time: Optional[int] = None,
    ):
        """Computes the timestamp after which the examples are left
        for validation"""
        if min_time is not None:
            where = f'WHERE {self.temporal_column} > {min_time}'
        else:
            where = ''
        return f'''
        SELECT
        quantileExact({quantile})({self.temporal_column})
        FROM {self.db_name}.{self.feature_table_name}
        {where}'''

    @clickhouse_transformation
    def get_examples(
        self,
        tables: List[str],
        columns: List[str],
        join_key: str,
        split: Literal['train', 'validation', 'test'],
        use_previous_downsamples: bool = True,
        limit: Optional[int] = None,
    ):
        """Fetches data from a clickhouse database into a pandas data frame

        Args:
          tables: Table names to be joined to 'main' table
          columns: Names of the columns to select after the join.
          joins_key: Name of the column to be used as join_key
          split: Data split to fetch.
          use_previous_downsamples: If we use previous downsampled tables or
            if we create the downsamples all over agin.
          limit: Integer to limit the number of examples to be extracted.
          

        """
        time_constraint = ''
        if split == 'train':
            if self.split_timestamp_train_bottom is not None:
                time_constraint += f'{self.db_name}.{self.feature_table_name}.{self.temporal_column} > {self.split_timestamp_train_bottom} AND'
                time_constraint += '\n'
            time_constraint += f'{self.db_name}.{self.feature_table_name}.{self.temporal_column} < {self.split_timestamp_train_top} AND'
            ds_ratio = self.ds_ratio_train
        elif split == 'validation':
            time_constraint += f'{self.db_name}.{self.feature_table_name}.{self.temporal_column} < {self.split_timestamp_val_top} AND'
            f'{self.db_name}.{self.feature_table_name}.{self.temporal_column} < {self.split_timestamp_val_top} AND'
            ds_ratio = self.ds_ratio_val
        else:
            if self.split_timestamp_test_top is not None:
                time_constraint += f'{self.db_name}.{self.feature_table_name}.{self.temporal_column} < {self.split_timestamp_test_top} AND'
                time_constraint += '\n'
            time_constraint = f'{self.db_name}.{self.feature_table_name}.{self.temporal_column} > {self.split_timestamp_test_bottom} AND'

        if not use_previous_downsamples:
            drop = f'''
            DROP DICTIONARY IF EXISTS {self.db_name}.dict_keys_{split};
            DROP TABLE IF EXISTS {self.db_name}.downsampled_{self.feature_table_name}_{split};
            DROP TABLE IF EXISTS {self.db_name}.downsampled_join_{split};'''
        else:
            drop = ''

        if split in ['train', 'validation']:
            selection_constraint =\
                f'({self.db_name}.{self.feature_table_name}.{self.label_column} = 1 OR '\
                f'({self.db_name}.{self.feature_table_name}.{self.label_column} = 0 AND '\
                f'randUniform(0, 1) > {1 - ds_ratio}))'
        else:
            selection_constraint = 'true'

        lim = f'ORDER BY rand() LIMIT {limit}' if limit is not None else ''

        select_keys = f'''
        CREATE TABLE IF NOT EXISTS {self.db_name}.downsampled_{self.feature_table_name}_{split}
        ENGINE=MergeTree ORDER BY {join_key}
        SETTINGS index_granularity = 8192, allow_nullable_key = 1
        AS (
            SELECT *
            FROM {self.db_name}.{self.feature_table_name}
            WHERE
            {time_constraint}
            {selection_constraint}
        );
        '''

        create_dict = f'''
        CREATE DICTIONARY IF NOT EXISTS {self.db_name}.dict_keys_{split}
        (
            join_key {self.join_key_type}
        )
        PRIMARY KEY {join_key}
        SOURCE(CLICKHOUSE(QUERY
           'SELECT {join_key}
            FROM {self.db_name}.downsampled_{self.feature_table_name}_{split}'))
        LIFETIME(MIN 600 MAX 900) LAYOUT(HASHED());

        '''

        join_tables = ''.join([
            f'JOIN {self.db_name}.{t} ON'\
            f' {self.db_name}.downsampled_{self.feature_table_name}_{split}.{join_key} ='
            f' {self.db_name}.{t}.{join_key}' + '\n'
            for t in tables
        ])
        dict_keys = ''.join([
            f'dictHas("{self.db_name}.dict_keys_{split}", {self.db_name}.{t}.{join_key}) AND'
            + '\n' for t in tables
        ] + ['true'])
        select_features = f'''
        CREATE TABLE IF NOT EXISTS {self.db_name}.downsampled_join_{split}
        ENGINE=MergeTree ORDER BY {join_key}
        SETTINGS index_granularity = 8192, allow_nullable_key = 1
        AS (
            SELECT {self.db_name}.downsampled_{self.feature_table_name}_{split}.{join_key} AS {join_key},
            *
            FROM {self.db_name}.downsampled_{self.feature_table_name}_{split}
            {join_tables}
            where
            {dict_keys}
        );

        SELECT {", ".join(columns)}
        FROM {self.db_name}.downsampled_join_{split}
        {lim};

        '''

        query = f'''
        {drop}
        {select_keys}
        {create_dict}
        {select_features}
        '''
        print(query)
        return query

    @clickhouse_transformation
    def _compute_selection_ratio(self, split: Literal['train', 'validation',
                                                      'test']):
        # TODO(augusto): Deprecated downsampling ration for 'test'
        # splits since we don't downsample.
        if split == 'train':
            where = f'WHERE {self.temporal_column} < {self.split_timestamp_train_top}'
            if self.split_timestamp_train_bottom is not None:
                where += ' AND '
                where += f'{self.temporal_column} > {self.split_timestamp_train_bottom}'
        elif split == 'validation':
            where = f'WHERE {self.temporal_column} > {self.split_timestamp_val_bottom} AND '\
                    f'{self.temporal_column} < {self.split_timestamp_val_top}'
        else:
            where = f'WHERE {self.temporal_column} > {self.split_timestamp_test_bottom}'
            if self.split_timestamp_test_top is not None:
                where += ' AND '
                where += f'{self.temporal_column} < {self.split_timestamp_test_top}'
        expr = f'{self.downsampling_rate} *'\
               f' sum({self.label_column})'\
               f'/ (count(*) - sum({self.label_column}))'
        return f'''
        SELECT LEAST(1, {expr})
        FROM {self.db_name}.{self.feature_table_name}
        {where}
        '''

    @clickhouse_transformation
    def _stats(self, split: Literal['train', 'validation', 'test']):
        if split == 'train':
            where = f'WHERE {self.temporal_column} < {self.split_timestamp_train_top}'
            if self.split_timestamp_train_bottom is not None:
                where += ' AND '
                where += f'{self.temporal_column} > {self.split_timestamp_train_bottom}'
        elif split == 'validation':
            where = f'WHERE {self.temporal_column} > {self.split_timestamp_val_bottom} AND '\
                    f'{self.temporal_column} < {self.split_timestamp_val_top}'
        else:
            where = f'WHERE {self.temporal_column} > {self.split_timestamp_test_bottom}'
            if self.split_timestamp_test_top is not None:
                where += ' AND '
                where += f'{self.temporal_column} < {self.split_timestamp_test_top}'

        return f'''
        SELECT SUM({self.label_column}) AS n_positive_samples,
               COUNT(*) - SUM({self.label_column}) AS n_negative_samples
        FROM {self.db_name}.{self.feature_table_name}
        {where}
        '''

    @clickhouse_transformation
    def show_tables(self):
        return f'''SHOW TABLES FROM {self.db_name}'''
