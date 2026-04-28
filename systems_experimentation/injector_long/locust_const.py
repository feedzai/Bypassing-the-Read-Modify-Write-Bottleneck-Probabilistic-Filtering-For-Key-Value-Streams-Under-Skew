import os
import pandas as pd
import logging

from locust import HttpUser, task, constant, events, constant_throughput
from locust.runners import WorkerRunner
from locust.exception import StopUser

DATA_FILE = "/app/data.parquet"
FILE_TYPE = "parquet"
TIME_COL = 'unix_seconds'
AMOUNT_COL = 'amount_usd'
ID_COL = 'merchant_id'
PAYLOAD_KEYS = [TIME_COL, AMOUNT_COL, ID_COL, 'amount_usd']

ENDPOINT = os.environ.get("LOAD_TEST_ENDPOINT", "")
WORKER_COUNT = int(os.environ.get("LOCUST_WORKERS", 1))
START_TIME_STAMP = 1420070400


def load_data(path):
    # Specify ONLY the columns needed for your calculations
    REQUIRED_COLS = ['Year', 'Month', 'Day', 'Time', 'Merchant Name', 'Amount']

    print(f'Loading data', flush=True)
    try:
        # We only bring the essential columns into memory
        df = pd.read_parquet(
            path,
            columns=REQUIRED_COLS,
            filters=[('Year', '>=', 2013)],
        )
        print('Loaded df into memory')
        date_cols = pd.DataFrame({
            'year':
            df['Year'],
            'month':
            df['Month'],
            'day':
            df['Day'],
            'hour':
            pd.to_numeric(df['Time'].str.split(':').str[0]),
            'minute':
            pd.to_numeric(df['Time'].str.split(':').str[1])
        })

        # 2. Convert to datetime directly (much lower memory footprint)
        df[TIME_COL] = pd.to_datetime(date_cols).astype('int64') // 10**9
        print('Created time stamp')

        # Once unix_seconds is created, drop Year/Month/Day/Time to free RAM
        df = df.drop(columns=['Year', 'Month', 'Day', 'Time'])
        print('Dropped extra cols')

        df[ID_COL] = df['Merchant Name']
        df['amount_usd'] = df['Amount'].astype(str).str.replace(
            r'[$,]', '', regex=True).astype(float)

        # Filter and Sort
        df = df[df[TIME_COL] > START_TIME_STAMP].sort_values(TIME_COL)

        return df
    except Exception as e:
        print(f"Failed: {e}", flush=True)
        return pd.DataFrame()


DF = load_data(DATA_FILE)
logging.info('Loaded data')
print('loaded the data')


class DataManager:
    _iterator = None

    @classmethod
    def load_data(cls, environment):
        if cls._iterator is not None:
            return None

        df = DF
        if isinstance(environment.runner, WorkerRunner):
            worker_id = environment.runner.worker_index
            df = DF.iloc[worker_id::WORKER_COUNT]
            logging.info(f"Worker {worker_id} processing {len(df)} rows.")

        cls._iterator = df.itertuples(index=False)

    @classmethod
    def get_next_event(cls):
        try:
            row = next(cls._iterator)

            return {
                key: getattr(row, key)
                for key in PAYLOAD_KEYS if hasattr(row, key)
            }
        except StopIteration:
            return None
        except Exception as e:
            print(f"Error reading row: {e}")
            return None


class HttpInjector(HttpUser):
    wait_time = constant_throughput(200)

    def on_start(self):
        DataManager.load_data(self.environment)

    @task
    def inject(self):
        payload = DataManager.get_next_event()

        if payload is None:
            print("Reached end")
            self.environment.runner.quit()
            self.stop()

        with self.client.post(ENDPOINT, json=payload,
                              catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed with status {response.status_code}")


@events.test_start.add_listener
def on_test_start(environment, **kwargs):
    DataManager.load_data(environment)
