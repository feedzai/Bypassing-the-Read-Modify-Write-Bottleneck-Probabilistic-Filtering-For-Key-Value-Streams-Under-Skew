"""Class that handles event filtering."""
from typing import List, Optional, Union
import tempfile
import os
import gc
import glob

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import math
import numba

from ..feature_profiles import (
    clickhouse_transformation,
    estimate_arrival_intensity,
    aggregation_over_window,
)
from smoothoperators import feature_profiles

STREAMING_SETTINGS = """
SETTINGS 
    max_threads = 1,
    max_insert_threads = 1,
    input_format_parallel_parsing = 0,
    max_insert_block_size = 10000,
    max_memory_usage = 4000000000  -- 4GB limit as a safety buffer
"""

_HF_TO_MS = {
    'minute': 60000,
    'hour': 60 * 60000,
    'day': 24 * 60 * 60000,
}

_MS_TO_HF = {
    60000: 'minute',
    60000 * 60: 'hour',
    60000 * 60 * 24: 'day',
}

import numba
import numpy as np


@numba.njit
def sigmoid(x):
    """Standard sigmoid function for p-adjustment."""
    return 1.0 / (1.0 + np.exp(-x))


@numba.njit
def _compute_stochastic_loop(times, quantities, half_life, threshold, norm_val,
                             var_alpha):
    n = len(times)
    intensities = np.zeros(n, dtype=np.float64)
    probs = np.zeros(n, dtype=np.float64)
    included = np.zeros(n, dtype=np.int8)

    # Initialize states
    curr_intensity_state = 0.0
    curr_q_sum_state = 0.0
    curr_q2_sum_state = 0.0

    last_time = times[0]
    decay_const = np.log(2) / half_life

    for i in range(n):
        t = times[i]
        q = quantities[i]
        dt = t - last_time

        decay_factor = np.exp(-decay_const * dt)
        curr_intensity_state *= decay_factor
        curr_q_sum_state *= decay_factor
        curr_q2_sum_state *= decay_factor

        p = min(1.0, (threshold * norm_val) / (curr_intensity_state + 1e-7))

        if var_alpha is not None and p < 1.0 and p > 0.0:
            if curr_intensity_state > 1e-7:
                mu = curr_q_sum_state / curr_intensity_state
                # Variance = E[X^2] - (E[X])^2
                raw_var = (curr_q2_sum_state / curr_intensity_state) - (mu**2)
                var = max(1e-7, raw_var)
            else:
                mu = q
                var = 1e-7

            logit_p = np.log(p / (1.0 - p))
            adjustment = var_alpha * (q - mu) / (np.sqrt(var) + 1e-7)
            p = sigmoid(logit_p + adjustment)

        r = np.random.rand()
        is_included = r < p

        intensities[i] = curr_intensity_state
        probs[i] = p
        included[i] = 1 if is_included else 0

        if is_included:
            inv_p = 1.0 / p
            curr_intensity_state += inv_p
            if var_alpha is not None:
                curr_q_sum_state += q * inv_p
                curr_q2_sum_state += (q**2) * inv_p

        last_time = t

    return intensities, probs, included


class DecayedSum:

    def __init__(self, halflife):
        self.lam = np.log(2) / halflife
        self.value = 0.0
        self.last_t = None
        self.n_updates = 0

    def update(self, t, q, p):
        if self.last_t is None:
            self.last_t = t
        dt = t - self.last_t
        self.value *= np.exp(-self.lam * dt)
        self.value += q / p
        self.last_t = t
        self.n_updates += 1

    def get(self, t, q):
        if self.last_t is None:
            dt = 0
        else:
            dt = t - self.last_t
        return self.value * np.exp(-self.lam * dt) + q


class DecayedMean:

    def __init__(self, half_life):
        self.count = DecayedSum(half_life)
        self.decayed_sum = DecayedSum(half_life)

    def update(self, t, q, p):
        self.count.update(t, 1, p)
        self.decayed_sum.update(t, q, p)

    def get(self, t, q):
        c = self.count.get(t, 1)
        if c == 0:
            return 0
        return self.decayed_sum.get(t, q) / c


class DecayedVariance:

    def __init__(self, half_life):
        self.mean = DecayedMean(half_life)
        self.mean_2 = DecayedMean(half_life)

    def update(self, t, q, p):
        self.mean.update(t, q, p)
        self.mean_2.update(t, q**2, p)

    def get(self, t, q):
        mean = self.mean.get(t, q)
        mean_2 = self.mean_2.get(t, q**2)
        return max(0, mean_2 - mean**2)


def custom_sci(x):
    """Formats floats as not to interfear with chdb syntax"""
    if x == 0:
        return "0e_00"
    exp = math.floor(math.log10(abs(x)))
    mantissa = x / (10**exp)
    while not mantissa.is_integer():
        mantissa *= 10
        exp -= 1
    sign = "_" if exp < 0 else ""
    return f"{int(mantissa)}e{sign}{abs(exp):02d}"


class EventFilter:
    """This class is responsible for simulating event filtering over our dataset.

    At a very high-level it works by:

    1. Computing intensity functions, i.e., event arrival rates for given
       entities. This is done using a decayed exponential moving count
       on the number of events grouped by entities.
    2. For given thresholds compute the probability of keeping the events with
       min(1, threshold/intensity)
    3. Created new tables with an included or not column.

    This class, then also has methods to compute the estimations of the original
    profiles from the filtered ones.

    Example usage:
        event_filter = event_filtering.EventFilter(
        'features',
        'events',
        'join_key',
        'unix_seconds',
        'event_time_unix',
    )

    with feature_profiles.ClickHouseConnector(args.session):
        event_filter.compute_intensity_functions(
            half_lifes=['minute', 'hour', 'day', 'week',],
            thresholds=[1, 2, 10],  # events per minute
            entity='merchant_id',
        )
        event_filter.simulate_filtering('merchant_id', 1)
        event_filter.compute_decayed_aggregations(
            [
                event_filtering.TimeDecayedCount('day'),
                event_filtering.TimeDecayedSum('day'),
                event_filtering.TimeDecayedAverage('day'),
            ],
            'merchant_id',
            'amount_usd',
        )

    """

    def __init__(
        self,
        db_name: str,
        event_table_name: str,
        join_key: str,
        unix_seconds: str,
        event_time_unix: str,
    ):
        """
        Args:
        db_name: Name of the database to use.
        event_table_name: Name of the table that contains the events
        join_key: Name of the column with the join keys
        unix_seconds: Name of the column with the time step in seconds. This
          is important for computing aggregations over rectangular windows
        event_time_unix: Name of the column with the timestamps of the events in
          miliseconds. This is used to get more accurate estimations for
          intensity functions.
        
        """
        self.db_name = db_name
        self.event_table_name = event_table_name
        self.join_key = join_key
        self.unix_seconds = unix_seconds
        self.event_time_unix = event_time_unix

        self.half_lifes = None
        self.thresholds = None
        self.num_trials = None

    def compute_intensity_functions(
        self,
        half_lifes: List[Union[float, str]],
        thresholds: List[int],
        entity: str,
        keys: Optional[List[str]] = None,
    ):
        """Creates a new table with intensity functions of each event for `entity`

        Args:
          half_lifes: The half lifes of the time decayed moving counts to be
            used for estimations
          thresholds: A list with the thresholds expressed in (events/minute).
          entity: The entity for which the intensity functions are estimated.
          keys: Optional list with extra keys to select and store in the created
            tables. Default behavior is [join_key, tevent_time_unix, unix_seconds
            , entity]

        """
        if keys is None:
            # Default behaviour is to select only the join key, the
            # time columns and entity, from the original table.
            keys = [
                self.join_key,
                self.event_time_unix,
                self.unix_seconds,
                entity,
            ]

        # Create a temp table ordered by the `entity` so that the
        # aggregation can be efficiently computed.
        self._group_by_table(self.event_table_name, entity)
        target_table_name = f'intensity_functions_{entity}'
        estimate_arrival_intensity(
            db_name=self.db_name,
            source_table_name=f'temp_{entity}_{self.event_table_name}',
            target_table_name=target_table_name,
            partition_by=entity,
            order_by=self.event_time_unix,
            half_lifes=half_lifes,
            select_keys=keys,
            col_names=[f'intensity_half_life_{hf}' for hf in half_lifes],
        )
        for hf in half_lifes:
            for t in thresholds:
                self._add_inclusion_probability_column(target_table_name, hf,
                                                       t)
        self.half_lifes = half_lifes
        self.thresholds = thresholds

    def simulate_filtering_double_stochastic(
        self,
        half_lifes: List[Union[float, str]],
        thresholds: List[int],
        entity: str,
        trials: int,
        quantity: str,
        var_alpha: Optional[float] = None,
    ):
        for trial in range(trials):
            with tempfile.TemporaryDirectory() as temp_dir:
                self.compute_intensity_functions_double_stochastic(
                    half_lifes,
                    thresholds,
                    entity,
                    trial,
                    temp_dir,
                    quantity,
                    var_alpha=var_alpha,
                )
        self.half_lifes = half_lifes
        self.thresholds = thresholds
        self.num_trials = trials

    def compute_intensity_functions_double_stochastic(
        self,
        half_lifes: List[Union[float, str]],
        thresholds: List[int],
        entity: str,
        trial: int,
        temp_dir,
        quantity,
        var_alpha: Optional[float] = None,
    ):
        """Double stochastic approach must be performed in python."""

        var_reduction = '' if var_alpha is None else f'_var_reduction_alpha_{custom_sci(var_alpha)}'
        incl_table = f'included_events_double_stochastic{var_reduction}_{entity}_trial_{trial}'
        int_table = f'intensity_functions_double_stochastic{var_reduction}_{entity}_trial_{trial}'
        # tables = feature_profiles.get_tables(self.db_name)['name'].values
        # if (int_table in tables) or (incl_table in tables):
        #     print('Skipping because table exists')
        #     return
        # else:
        print(f'Creating table {incl_table}')
        print(f'Creating table {int_table}')
        self._create_persistent_tables(incl_table, int_table, half_lifes,
                                       thresholds, entity)
        ids = self.fetch_entity_ids(entity)[entity].values
        num_entities = len(ids)
        entities_in_table = self.fetch_entity_ids(entity,
                                                  incl_table)[entity].values
        for i, entity_id in enumerate(ids):
            if entity_id not in entities_in_table:
                events = self.fetch_entity(entity_id, entity, quantity)
                print(f'running for entity {i+1} out of {num_entities}')
                self._double_stochastic_intensity_functions(
                    events,
                    half_lifes,
                    thresholds,
                    entity,
                    entity_id,
                    temp_dir,
                    quantity,
                    var_alpha=var_alpha,
                )
                del events
                gc.collect()

                for f in glob.glob(f'{temp_dir}/included_*.parquet'):
                    print(f'inserting {f} into tables')
                    self._insert_file_into_table(f, incl_table, int_table,
                                                 half_lifes, thresholds,
                                                 entity)
                    os.remove(f)
            else:
                print(f'Skipping entity {entity_id}. Already in table')
            # self._create_or_insert_into_tables(
            #     f,
            #     half_lifes,
            #     thresholds,
            #     entity,
            #     trial,
            #     var_alpha,
            # )
    @clickhouse_transformation
    def _create_persistent_tables(self, incl_table, int_table, half_lifes,
                                  thresholds, entity):
        """Defines the schema explicitly to avoid memory-heavy inference."""

        # Build Included Schema
        incl_cols = [
            f"join_key UInt64, {entity} String, {self.event_time_unix} UInt64"
        ]
        for hf in half_lifes:
            for ts in thresholds:
                incl_cols.append(
                    f"half_life_{hf}_threshold_{custom_sci(ts)}_included Int8")
                incl_cols.append(
                    f"p_half_life_{hf}_threshold_{custom_sci(ts)} Float64")

        # Build Intensity Schema
        int_cols = [
            f"join_key UInt64, {entity} String, {self.event_time_unix} UInt64"
        ]
        for hf in half_lifes:
            for ts in thresholds:
                int_cols.append(
                    f"intensity_half_life_{hf}_threshold_{custom_sci(ts)} Float64"
                )

        return f"""
        CREATE TABLE IF NOT EXISTS {self.db_name}.{incl_table} ({", ".join(incl_cols)})
        ENGINE = MergeTree ORDER BY ({entity}, {self.event_time_unix})
        SETTINGS index_granularity = 8192;

        CREATE TABLE IF NOT EXISTS {self.db_name}.{int_table} ({", ".join(int_cols)})
        ENGINE = MergeTree ORDER BY ({entity}, {self.event_time_unix})
        SETTINGS index_granularity = 8192;
        """

    @clickhouse_transformation
    def _insert_file_into_table(self, f, incl_table, int_table, half_lifes,
                                thresholds, entity):
        """Performs a memory-safe streaming insert."""

        # Map columns for the SELECT statement (matching your assumeNotNull logic)
        if 'included' in f:
            target_table = incl_table
            cols = ["join_key", entity, self.event_time_unix]
            for hf in half_lifes:
                for ts in thresholds:
                    cols.extend([
                        f"half_life_{hf}_threshold_{custom_sci(ts)}_included",
                        f"p_half_life_{hf}_threshold_{custom_sci(ts)}"
                    ])
        else:
            target_table = int_table
            cols = ["join_key", entity, self.event_time_unix]
            for hf in half_lifes:
                for ts in thresholds:
                    cols.append(
                        f"intensity_half_life_{hf}_threshold_{custom_sci(ts)}")

        # Clean column list and add Null check if necessary
        select_cols = ", ".join([f"assumeNotNull({c})" for c in cols])

        return f"""
        INSERT INTO {self.db_name}.{target_table}
        {STREAMING_SETTINGS}
        SELECT {select_cols} FROM file('{f}', Parquet);
        """

    # @clickhouse_transformation
    # def _create_or_insert_into_tables(
    #     self,
    #     f: str,
    #     half_lifes: List[Union[float, str]],
    #     thresholds: List[int],
    #     entity: str,
    #     trial: int,
    #     var_alpha: Optional[float] = None,
    # ):
    #     """Creates the table from a single file"""
    #     cols_included = [
    #         f'half_life_{half_life}_threshold_{custom_sci(threshold)}_included'
    #         for half_life in half_lifes for threshold in thresholds
    #     ]
    #     cols_included += [
    #         f'p_half_life_{half_life}_threshold_{custom_sci(threshold)}'
    #         for half_life in half_lifes for threshold in thresholds
    #     ] + [self.join_key, entity, self.event_time_unix]
    #     cols_included = [
    #         f'assumeNotNull({exp}) AS {exp}' for exp in cols_included
    #     ]

    #     intensity_included = [
    #         f'intensity_half_life_{half_life}_threshold_{custom_sci(threshold)}'
    #         for half_life in half_lifes for threshold in thresholds
    #     ] + [self.join_key, entity, self.event_time_unix]
    #     intensity_included = [
    #         f'assumeNotNull({exp}) AS {exp}' for exp in intensity_included
    #     ]
    #     var_reduction = '' if var_alpha is None else f'_var_reduction_alpha_{custom_sci(var_alpha)}'
    #     incl_table = f'{self.db_name}.included_events_double_stochastic{var_reduction}_{entity}_trial_{trial}'
    #     int_table = f'{self.db_name}.intensity_functions_double_stochastic{var_reduction}_{entity}_trial_{trial}'

    #     mem_settings = """
    #     SETTINGS
    #         max_memory_usage = 0,
    #         max_insert_threads = 4,
    #         max_threads = 4,
    #         max_insert_block_size = 100000,
    #         max_execution_time = 0"""
    #     if 'included' in f:
    #         return f'''
    #     -- 1. Create Included Table (Schema only)
    #     CREATE TABLE IF NOT EXISTS {incl_table}
    #     ENGINE = MergeTree
    #     ORDER BY ({entity}, {self.event_time_unix})
    #     SETTINGS index_granularity = 8192, allow_nullable_key = 1
    #     AS SELECT {",".join(cols_included)}
    #     FROM file(\'{f}\', Parquet) LIMIT 0;

    #     -- 2. Insert Data into Included Table
    #     INSERT INTO {incl_table}
    #     {mem_settings}
    #     SELECT {",".join(cols_included)}
    #     FROM file(\'{f}\', Parquet);

    #     '''
    #     return f'''
    #     -- 3. Create Intensity Table (Schema only)
    #     CREATE TABLE IF NOT EXISTS {int_table}
    #     ENGINE = MergeTree
    #     ORDER BY ({entity}, {self.event_time_unix})
    #     SETTINGS index_granularity = 8192, allow_nullable_key = 1
    #     AS SELECT {",".join(intensity_included)}
    #     FROM file(\'{f}\', Parquet) LIMIT 0;

    #     -- 4. Insert Data into Intensity Table
    #     INSERT INTO {int_table}
    #     {mem_settings}
    #     SELECT {",".join(intensity_included)}
    #     FROM file(\'{f}\', Parquet);
    #     '''

    def _double_stochastic_intensity_functions(self,
                                               data_frame,
                                               half_lifes,
                                               thresholds,
                                               entity,
                                               entity_id,
                                               save_dir,
                                               quantity,
                                               var_alpha=None):
        # 1. Convert DF columns to NumPy arrays ONCE (Massive memory saving)
        times = data_frame[self.event_time_unix].values.astype(np.uint64)
        quantities = data_frame[quantity].values.astype(np.float64)
        join_keys = data_frame[self.join_key].values.astype(np.uint64)
        entities = data_frame[entity].astype(str).values

        intensities_results = {}
        included_results = {}
        probs_results = {}

        norm = {
            'minute': 1,
            'hour': 60,
            'day': 60 * 24,
            'week': 60 * 24 * 7,
            'month': 60 * 24 * 7 * 30
        }
        # 2. Run the math via the JIT kernel
        for hf_key in half_lifes:
            hf_ms = _HF_TO_MS.get(hf_key, hf_key)
            for ts in thresholds:
                # Call the Numba function
                ints, prbs, incl = _compute_stochastic_loop(
                    times, quantities, hf_ms, ts, norm[hf_key], var_alpha)

                # Store results in dictionaries
                col_suffix = f"half_life_{hf_key}_threshold_{custom_sci(ts)}"
                intensities_results[f"intensity_{col_suffix}"] = ints
                included_results[f"{col_suffix}_included"] = incl
                probs_results[f"p_{col_suffix}"] = prbs

        # 3. Assemble Arrow Tables (Outside JIT)
        # Avoid the {**d1, **d2} syntax that caused your error
        int_data = {
            'join_key': join_keys,
            entity: entities,
            self.event_time_unix: times
        }
        int_data.update(intensities_results)

        int_names = ['join_key', entity, self.event_time_unix]
        int_arrays = [
            pa.array(join_keys, type=pa.uint64()),
            pa.array(entities, type=pa.string()),
            pa.array(times, type=pa.uint64())
        ]

        for k, v in intensities_results.items():
            int_names.append(k)
            int_arrays.append(pa.array(v, type=pa.float64()))

        intensity_table = pa.Table.from_arrays(int_arrays, names=int_names)

        pq.write_table(
            intensity_table,
            os.path.join(save_dir, f'intensity_{entity_id}.parquet'),
            # Smaller row groups help ClickHouse stream this file back in with less RAM
            row_group_size=50000)

        # --- 4. Assemble and Write Included Table ---
        incl_names = ['join_key', entity, self.event_time_unix]
        incl_arrays = [
            pa.array(join_keys, type=pa.uint64()),
            pa.array(entities, type=pa.string()),
            pa.array(times, type=pa.uint64())
        ]

        # Add the 'included' binary flags (Int8)
        for k, v in included_results.items():
            incl_names.append(k)
            incl_arrays.append(pa.array(v, type=pa.int8()))

        # Add the probabilities (Float64)
        for k, v in probs_results.items():
            incl_names.append(k)
            incl_arrays.append(pa.array(v, type=pa.float64()))

        included_table = pa.Table.from_arrays(incl_arrays, names=incl_names)

        pq.write_table(included_table,
                       os.path.join(save_dir, f'included_{entity_id}.parquet'),
                       row_group_size=50000)

    # @numba.jit
    # def _double_stochastic_intensity_functions(
    #         self,
    #         data_frame,
    #         half_lifes,
    #         thresholds,
    #         entity,
    #         entity_id,
    #         save_dir,
    #         quantity,
    #         var_alpha: Optional[float] = None):
    #     intensities_events = {}
    #     included_events = {}
    #     probs_events = {}
    #     norm = {
    #         'minute': 1,
    #         'hour': 60,
    #         'day': 60 * 24,
    #         'week': 60 * 24 * 7,
    #         'month': 60 * 24 * 7 * 30
    #     }
    #     for half_life in [_HF_TO_MS.get(hf, hf) for hf in half_lifes]:
    #         for threshold in thresholds:
    #             intensity = DecayedSum(half_life)
    #             mean = DecayedMean(half_life)
    #             variance = DecayedVariance(half_life)
    #             intensities = []
    #             probs = []
    #             included = []
    #             for _, row in data_frame.iterrows():
    #                 current_time = row[self.event_time_unix]
    #                 curr_intensity = intensity.get(current_time, 1)
    #                 p = min(
    #                     1, threshold * norm[_MS_TO_HF[half_life]] /
    #                     (curr_intensity + 1e-7))
    #                 q = row[quantity]
    #                 if var_alpha is not None and p < 1:
    #                     mu = mean.get(current_time, q)
    #                     var = variance.get(current_time, q)
    #                     p = sigmoid(
    #                         np.log((p) / (1 - p)) + var_alpha * (q - mu) /
    #                         (np.sqrt(var) + 1e-7))
    #                 r = np.random.rand()
    #                 if r < p:
    #                     intensity.update(current_time, 1, p)
    #                     if var_alpha is not None:
    #                         mean.update(current_time, q, p)
    #                         variance.update(current_time, q, p)

    #                 intensities.append(curr_intensity)
    #                 probs.append(p)
    #                 included.append(int(r < p))

    #             intensities_events[
    #                 f'intensity_half_life_{_MS_TO_HF.get(half_life, half_life)}_threshold_{custom_sci(threshold)}'] = intensities
    #             included_events[
    #                 f'half_life_{_MS_TO_HF.get(half_life, half_life)}_threshold_{custom_sci(threshold)}_included'] = included
    #             probs_events[
    #                 f'p_half_life_{_MS_TO_HF.get(half_life, half_life)}_threshold_{custom_sci(threshold)}'] = probs

    #     intensities = pa.Table.from_pandas(
    #         pd.DataFrame(
    #             {
    #                 **{
    #                     'join_key':
    #                     data_frame[self.join_key].values,
    #                     entity:
    #                     data_frame[entity].astype(str).values,
    #                     self.event_time_unix:
    #                     data_frame[self.event_time_unix].values,
    #                 },
    #                 **intensities_events
    #             }),
    #         preserve_index=False,
    #         schema=pa.schema([
    #             ('join_key', pa.uint64()),
    #             (entity, pa.string()),
    #             (self.event_time_unix, pa.uint64()),
    #         ] + [(k, pa.float64()) for k in intensities_events.keys()]))

    #     pq.write_table(
    #         intensities,
    #         os.path.join(save_dir, f'intensity_{entity_id}.parquet'))

    #     included = pa.Table.from_pandas(
    #         pd.DataFrame(
    #             {
    #                 **{
    #                     'join_key':
    #                     data_frame[self.join_key].values,
    #                     entity:
    #                     data_frame[entity].astype(str).values,
    #                     self.event_time_unix:
    #                     data_frame[self.event_time_unix].values,
    #                 },
    #                 **included_events,
    #                 **probs_events,
    #             }),
    #         preserve_index=False,
    #         schema=pa.schema([
    #             ('join_key', pa.uint64()),
    #             (entity, pa.string()),
    #             (self.event_time_unix, pa.uint64()),
    #         ] + [(k, pa.int8()) for k in included_events.keys()] +
    #                          [(k, pa.float64()) for k in probs_events.keys()]))

    #     pq.write_table(included,
    #                    os.path.join(save_dir, f'included_{entity_id}.parquet'))

    @clickhouse_transformation
    def table_size(self):
        return f'''
        SELECT 
            table,
            sum(bytes_on_disk) AS total_bytes,
            formatReadableSize(sum(bytes_on_disk)) AS total_size
        FROM system.parts
        WHERE database = '{self.db_name}'
          AND table = '{self.event_table_name}'
          AND active = 1
        GROUP BY table;
        '''

    @clickhouse_transformation
    def fetch_entity(self, entity_id, entity, quantity):
        return f'''
        SELECT {self.join_key},
               {self.event_time_unix},
               {entity},
               {quantity},
        FROM {self.db_name}.{self.event_table_name}
        WHERE {entity} = \'{entity_id}\'
        ORDER BY {self.event_time_unix};
        '''

    @clickhouse_transformation
    def fetch_entity_ids(self, entity, table=None):
        """Fetches event table from clickhouse to compute intensity for ds approach."""
        if table is None:
            table = self.event_table_name
        return f'SELECT DISTINCT {entity} FROM {self.db_name}.{table}'

    @clickhouse_transformation
    def _add_inclusion_probability_column(self, table_name, half_life,
                                          threshold):
        """Helper function to compute inclusion probabilities"""
        norm = {
            'minute': 1,
            'hour': 60,
            'day': 60 * 24,
            'week': 60 * 24 * 7,
            'month': 60 * 24 * 7 * 30
        }
        new_col_name = f'p_half_life_{half_life}_threshold_{custom_sci(threshold)}'
        expression = f'least(1, ({threshold} / intensity_half_life_{half_life}) * {norm[half_life]})'
        return f'''
        ALTER TABLE {self.db_name}.{table_name}
        ADD COLUMN IF NOT EXISTS {new_col_name}  Float32
        DEFAULT {expression};'''

    def simulate_filtering(
        self,
        entity,
        num_trials,
    ):
        """Simulates filtering `num_trial` times."""
        for i in range(num_trials):
            print(f'Filtering events for trial {i}...')
            self._filter(entity, i)
        self.num_trials = num_trials

    @clickhouse_transformation
    def _filter(
        self,
        entity,
        trial,
    ):
        """Creates a table with the events that are filtered This is achieved with:

           SELECT randUniform(0, 1) < p_half_life_{hf}_threshold_{t} AS
                  half_life_{hf}_threshold_{t}_included

        Which stores if the event is included or not.

        """
        table_name = f'intensity_functions_{entity}'
        included = []
        for hf in self.half_lifes:
            for t in self.thresholds:
                expression = f'randUniform(0, 1) < p_half_life_{hf}_threshold_{custom_sci(t)} '
                expression += f'AS half_life_{hf}_threshold_{custom_sci(t)}_included '
                included.append(expression)
                included.append(f'p_half_life_{hf}_threshold_{custom_sci(t)}')
        join = f'{self.db_name}.{table_name} AS E'
        return f'''
        CREATE TABLE IF NOT EXISTS
        {self.db_name}.included_events_{entity}_trial_{trial}
        ENGINE=MergeTree
        ORDER BY ({entity}, {self.event_time_unix})
        SETTINGS index_granularity = 8192, allow_nullable_key = 1
        AS(
            SELECT E.{self.join_key} as {self.join_key},
                   E.{entity} as {entity},
                   E.{self.event_time_unix} as {self.event_time_unix},
                   E.{self.unix_seconds} as {self.unix_seconds},
                   {(",").join(included)}
            FROM {join}
        )'''

    def compute_rectangular_aggregations(
        self,
        windowed_aggregations,
        window_bottoms,
        entity,
        quantity,
    ):
        print('Sorting event table...')
        self._group_by_table(self.event_table_name, entity)
        self._unfiltered_rectangular_aggregations(windowed_aggregations,
                                                  window_bottoms, entity,
                                                  quantity)
        for i in range(self.num_trials):
            print(f'Computing aggregations for trial {i}')
            self._rectangular_aggregations_trial(windowed_aggregations,
                                                 window_bottoms, entity,
                                                 quantity, i)

    def compute_decayed_aggregations(
        self,
        aggregations,
        entity,
        quantity,
        double_stochastic=False,
        drop_table=False,
        var_alpha: Optional[float] = None,
    ):
        """Computes exponentially decayed aggregations.

        Args:
          aggregations: List with aggregation classes, e.g, TimeDecayedSum('day'),
          entity: The entity for which to compute the aggregations
          quantity: The column with the quantity of interest, e.g., amount_usd.

        """
        print('Sorting event table...')
        self._group_by_table(self.event_table_name, entity)
        self._unfiltered_decayed_aggregations(aggregations, entity, quantity)
        for i in range(self.num_trials):
            print(f'Computing aggregations for trial {i}')
            self._decayed_aggregations_trial(
                aggregations,
                entity,
                quantity,
                i,
                double_stochastic=double_stochastic,
                drop_table=drop_table,
                var_alpha=var_alpha)

    def _unfiltered_decayed_aggregations(
        self,
        aggregations,
        entity,
        quantity,
    ):
        "Computes the unfiltered estimations"
        self._group_by_table(self.event_table_name, entity)
        source_table_name = f'temp_{entity}_{self.event_table_name}'
        for agg in aggregations:
            expressions = []
            col_names = []
            target_table_name = f'{str(agg)}_{quantity}_{entity}'
            expressions.append(
                agg.unfiltered.format(
                    **{
                        'column_name': quantity,
                        'time': self.event_time_unix,
                        'quantity': quantity,
                    }))
            col_names.append(f'{str(agg)}_{quantity}')
            aggregation_over_window(
                db_name=self.db_name,
                source_table_name=source_table_name,
                target_table_name=target_table_name,
                partition_by=[entity],
                order_by=[self.event_time_unix],
                order_by_windows=[self.event_time_unix],
                window_type=['rows'],
                window_bottom=['UNBOUNDED'],
                window_top=[None],
                select_keys=[
                    self.join_key,
                    entity,
                    self.unix_seconds,
                    self.event_time_unix,
                ],
                aggregation_expressions=expressions,
                col_names=col_names,
                units='miliseconds',
            )

    def _decayed_aggregations_trial(
        self,
        aggregations,
        entity,
        quantity,
        trial,
        double_stochastic=False,
        drop_table=False,
        var_alpha: Optional[float] = None,
    ):
        source_table_name = f'temp_{entity}_{self.event_table_name} AS E JOIN '
        if not double_stochastic:
            source_table_name +=\
            f'{self.db_name}.included_events_{entity}_trial_{trial} AS I '
        else:
            var_reduction = '' if var_alpha is None else f'_var_reduction_alpha_{custom_sci(var_alpha)}'
            source_table_name +=\
            f'{self.db_name}.included_events_double_stochastic{var_reduction}_{entity}_trial_{trial} AS I '
        source_table_name += f'ON E.{self.join_key} = I.{self.join_key}'

        for agg in aggregations:
            print(str(agg))
            for hf in self.half_lifes:
                for t in self.thresholds:
                    expressions = []
                    col_names = []
                    if not double_stochastic:
                        target_table_name = f'{agg}_{quantity}_{entity}_hf_{hf}_threshold_{custom_sci(t)}_{trial}'
                    else:
                        var_reduction = '' if var_alpha is None else f'_var_reduction_alpha_{custom_sci(var_alpha)}'
                        target_table_name = f'double_stochastic{var_reduction}_{agg}_{quantity}_{entity}_hf_{hf}_threshold_{custom_sci(t)}_{trial}'
                    if drop_table:
                        feature_profiles.drop_table(self.db_name,
                                                    target_table_name)
                    # Filtered estimations
                    expressions.append(
                        agg.filtered.format(
                            **{
                                'in_prob':
                                f'I.p_half_life_{hf}_threshold_{custom_sci(t)}',
                                'included':
                                f'I.half_life_{hf}_threshold_{custom_sci(t)}_included',
                                'time': self.event_time_unix,
                                'quantity': f'E.{quantity}',
                            }))
                    col_names.append(
                        f'{str(agg)}_{quantity}_half_life_{hf}_threshold_{custom_sci(t)}'
                    )

                    # Variance of the estimations
                    if agg.variance is not None:
                        expressions.append(
                            agg.variance.format(
                                **{
                                    'in_prob':
                                    f'I.p_half_life_{hf}_threshold_{custom_sci(t)}',
                                    'included':
                                    f'I.half_life_{hf}_threshold_{custom_sci(t)}_included',
                                    'time': self.event_time_unix,
                                    'quantity': f'E.{quantity}',
                                }))

                        col_names.append(
                            f'var_{str(agg)}_{quantity}_half_life_{hf}_threshold_{custom_sci(t)}'
                        )

                    aggregation_over_window(
                        db_name=self.db_name,
                        source_table_name=source_table_name,
                        target_table_name=target_table_name,
                        partition_by=[entity],
                        order_by=[self.event_time_unix],
                        order_by_windows=[self.event_time_unix],
                        window_type=['rows'],
                        window_bottom=['UNBOUNDED'],
                        window_top=[None],
                        select_keys=[
                            self.join_key,
                            entity,
                            self.unix_seconds,
                            self.event_time_unix,
                        ],
                        aggregation_expressions=expressions,
                        col_names=col_names,
                        units='miliseconds',
                    )

    @clickhouse_transformation
    def kept_volume(
        self,
        entity,
        trial,
        double_stochastic=False,
        var_alpha: Optional[float] = None,
    ):
        """Computes the volume, for a trial that was kept in the dataset."""
        select_keys = []
        for hf in self.half_lifes:
            for t in self.thresholds:
                if not double_stochastic:
                    select_keys.append(
                        f'AVG(half_life_{hf}_threshold_{custom_sci(t)}_included) AS kept_volume_half_life_{hf}_threshold_{custom_sci(t)}'
                    )
                else:
                    select_keys.append(
                        f'AVG(half_life_{hf}_threshold_{custom_sci(t)}_included) AS kept_volume_half_life_{hf}_threshold_{custom_sci(t)}'
                    )
        if not double_stochastic:
            from_clause = f'included_events_{entity}_trial_{trial}'
        else:
            var_reduction = '' if var_alpha is None else f'_var_reduction_alpha_{custom_sci(var_alpha)}'
            from_clause = f'included_events_double_stochastic{var_reduction}_{entity}_trial_{trial}'
        query = f'''
        SELECT {", ".join(select_keys)}
        FROM {self.db_name}.{from_clause}
        '''
        return query

    @clickhouse_transformation
    def _get_tables(self):
        return f'SHOW TABLES FROM {self.db_name}'

    @clickhouse_transformation
    def _group_by_table(self, table_name, entity):
        """Creates an aux table sorted by entity to then perform aggregations."""
        print('Ordering table...')
        return f'''
        CREATE TABLE IF NOT EXISTS {self.db_name}.temp_{entity}_{table_name}
        ENGINE=MergeTree
        ORDER BY ({entity}, {self.event_time_unix})
        SETTINGS index_granularity = 8192, allow_nullable_key = 1
        AS (
            SELECT *,
            {self.event_time_unix} / 1000 AS {self.unix_seconds}
            FROM {self.db_name}.{table_name}
        )'''


class RandomFilter:
    """Filter events with constant probability according to a given
    threshold.


    Example usage:

    >>> kept_volumes = event_filter.kept_volume(args.entity,
    ...                                                 2).loc[0].to_dict()
    >>> random_filter = event_filtering.RandomFilter(
    ...                        'features',
    ...                        'events',
    ...                        'join_key',
    ...                        'unix_seconds',
    ...                        'event_time_unix',
    ...                        )
    >>> random_filter.simulate_filtering_at_volumes(1, [.1, .2])

    This is simulate random filtering such that 90% and 80% of the dataset
    will be included:

    included_events_random_trial_1:

    join_key | included_at_volume_1e_1 | p_included_at_volume_1e_1 | included_at_volume_2e_1 | p_included_at_volume_2e_1 |
    1234     | 0                       | .1                        | 1                       | .2                        |
    1235     | 0                       | .1                        | 0                       | .2                        |
    1236     | 0                       | .1                        | 0                       | .2                        |
    1237     | 0                       | .1                        | 0                       | .2                        |
    1238     | 1                       | .1                        | 1                       | .2                        |

    The aggregations are then computed over these filtered profiles.
    """

    def __init__(
        self,
        db_name: str,
        event_table_name: str,
        join_key: str,
        unix_seconds: str,
        event_time_unix: str,
    ):
        """
        Args:
        db_name: Name of the database to use.
        event_table_name: Name of the table that contains the events
        join_key: Name of the column with the join keys
        unix_seconds: Name of the column with the time step in seconds. This
          is important for computing aggregations over rectangular windows
        event_time_unix: Name of the column with the timestamps of the events in
          miliseconds. This is used to get more accurate estimations for
          intensity functions.

        """
        self.db_name = db_name
        self.event_table_name = event_table_name
        self.join_key = join_key
        self.unix_seconds = unix_seconds
        self.event_time_unix = event_time_unix

        self.num_trials = None
        self.kept_volumes = None

    def simulate_filtering_at_volumes(self, num_trials: int,
                                      kept_volumes: List[float]):
        """Run num_trials event filtering simulations at all kept_volumes"""
        self.num_trials = num_trials
        self.kept_volumes = kept_volumes
        for t in range(num_trials):
            self.simulate_filtering_at_volumes_trial(kept_volumes, t)

    @clickhouse_transformation
    def simulate_filtering_at_volumes_trial(self, kept_volumes: List[float],
                                            trial: int):
        """Filters events at a given kept volume. The inclusion
        probability is equal to the volume of the data we want to
        keep."""

        included = []
        for kept_volume in kept_volumes:
            expression = f'\nrandUniform(0, 1) < {kept_volume}'
            expression += f' AS included_at_volume_{str(custom_sci(kept_volume))} '
            included.append(expression)
            included.append(
                f'\n{kept_volume} AS p_included_at_volume_{str(custom_sci(kept_volume))} '
            )

        return f'''
        CREATE TABLE IF NOT EXISTS
        {self.db_name}.included_events_random_trial_{trial}
        ENGINE=MergeTree
        ORDER BY ({self.event_time_unix})
        SETTINGS index_granularity = 8192, allow_nullable_key = 1
        AS(
            SELECT {self.join_key},
                   {self.event_time_unix},
                   {(",").join(included)}
            FROM {self.db_name}.{self.event_table_name}
        )'''

    def compute_decayed_aggregations(
        self,
        aggregations,
        entity,
        quantity,
    ):
        """Computes exponentially decayed aggregations.

        Args:
          aggregations: List with aggregation classes, e.g, TimeDecayedSum('day'),
          entity: The entity for which to compute the aggregations
          quantity: The column with the quantity of interest, e.g., amount_usd.

        """
        if self.num_trials is None:
            raise ValueError('Number of trials cannot be none.')
        print('Sorting event table...')
        self._group_by_table(self.event_table_name, entity)
        for i in range(self.num_trials):
            print(f'Computing aggregations for trial {i}')
            self._decayed_aggregations_trial(aggregations, entity, quantity, i)

    def _decayed_aggregations_trial(
        self,
        aggregations,
        entity,
        quantity,
        trial,
    ):
        if self.kept_volumes is None:
            raise ValueError('Kept volumes cannot be none')

        source_table_name = f'temp_{entity}_{self.event_table_name} AS E JOIN '
        source_table_name +=\
            f'{self.db_name}.included_events_random_trial_{trial} AS I '
        source_table_name += f'ON E.{self.join_key} = I.{self.join_key}'

        for agg in aggregations:
            for volume in self.kept_volumes:
                expressions = []
                col_names = []
                target_table_name = f'{agg}_{quantity}_{entity}_at_volume_{custom_sci(volume)}_{trial}'
                print(target_table_name)
                # Filtered estimations
                expressions.append(
                    agg.filtered.format(
                        **{
                            'in_prob':
                            f'I.p_included_at_volume_{custom_sci(volume)}',
                            'included':
                            f'I.included_at_volume_{custom_sci(volume)}',
                            'time': self.event_time_unix,
                            'quantity': f'E.{quantity}',
                        }))
                col_names.append(
                    f'{str(agg)}_{quantity}_at_volume_{custom_sci(volume)}')

                # Variance of the estimations
                if agg.variance is not None:
                    expressions.append(
                        agg.variance.format(
                            **{
                                'in_prob':
                                f'I.p_included_at_volume_{custom_sci(volume)}',
                                'included':
                                f'I.included_at_volume_{custom_sci(volume)}',
                                'time': self.event_time_unix,
                                'quantity': f'E.{quantity}',
                            }))

                    col_names.append(
                        f'var_{str(agg)}_{quantity}_at_volume_{custom_sci(volume)}'
                    )

                aggregation_over_window(
                    db_name=self.db_name,
                    source_table_name=source_table_name,
                    target_table_name=target_table_name,
                    partition_by=[entity],
                    order_by=[self.event_time_unix],
                    order_by_windows=[self.event_time_unix],
                    window_type=['rows'],
                    window_bottom=['UNBOUNDED'],
                    window_top=[None],
                    select_keys=[
                        self.join_key,
                        entity,
                        self.unix_seconds,
                        self.event_time_unix,
                    ],
                    aggregation_expressions=expressions,
                    col_names=col_names,
                    units='miliseconds',
                )

    @clickhouse_transformation
    def _group_by_table(self, table_name, entity):
        """Creates an aux table sorted by entity to then perform aggregations."""
        print('Ordering table...')
        return f'''
        CREATE TABLE IF NOT EXISTS {self.db_name}.temp_{entity}_{table_name}
        ENGINE=MergeTree
        ORDER BY ({entity}, {self.event_time_unix})
        SETTINGS index_granularity = 8192, allow_nullable_key = 1
        AS (
            SELECT *,
            {self.event_time_unix} / 1000 AS {self.unix_seconds}
            FROM {self.db_name}.{table_name}
        )'''

    @clickhouse_transformation
    def kept_volume(self, trial):
        """Computes the volume, for a trial that was kept in the dataset."""
        if self.kept_volumes is None:
            raise ValueError('Kept volumes cannot be none')
        select_keys = []
        for volume in self.kept_volumes:
            select_keys.append(
                f'\nAVG(included_at_volume_{custom_sci(volume)}) AS kept_volume_at_volume_{custom_sci(volume)}'
            )
        query = f'''
        SELECT {", ".join(select_keys)}
        FROM {self.db_name}.included_events_random_trial_{trial}
        '''
        return query
