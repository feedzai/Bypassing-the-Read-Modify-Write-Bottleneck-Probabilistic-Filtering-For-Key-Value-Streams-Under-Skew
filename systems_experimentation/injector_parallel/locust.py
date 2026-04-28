import os
import hashlib
import pandas as pd
from locust import FastHttpUser, task, constant
from locust.exception import StopUser

# Define the global worker count (defaulting to 4)
# NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 4))
NUM_WORKERS = int(os.environ.get("NUM_WORKERS", 1))

# ... (Keep your existing DATA_FILE, load_data, and iterator logic exact the same) ...
DATA_FILE = os.path.expanduser("/app/data.parquet")
# DATA_FILE = os.path.expanduser("~/Documents/ibm-credit-card-transactions/data.parquet")
ENDPOINT = os.environ.get("LOAD_TEST_ENDPOINT", "/api/v1/filtered")
# ENDPOINT = os.environ.get("LOAD_TEST_ENDPOINT", "/api/v1/unfiltered")
START_TIME_STAMP = 1420070400
PAYLOAD_KEYS = ['unix_seconds', 'amount_usd', 'merchant_id']


def load_data(path):
    print(f'Loading data from {path}...', flush=True)
    try:
        df = pd.read_parquet(
            path,
            columns=[
                'Year', 'Month', 'Day', 'Time', 'Merchant Name', 'Amount'
            ],
            filters=[('Year', '>=', 2013)],
        )

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

        df['unix_seconds'] = pd.to_datetime(date_cols).astype('int64') // 10**9
        df['merchant_id'] = df['Merchant Name']
        df['amount_usd'] = df['Amount'].astype(str).str.replace(
            r'[$,]', '', regex=True).astype(float)

        df = df[df['unix_seconds'] > START_TIME_STAMP].sort_values(
            'unix_seconds')
        print(f"Data load complete. {len(df)} rows ready for injection.",
              flush=True)
        return df
    except Exception as e:
        print(f"Failed to load data: {e}", flush=True)
        return pd.DataFrame()


# 1. Load data once into memory when the module initializes
DF = load_data(DATA_FILE)

# 2. Create a single, shared iterator.
# This guarantees no two Locust users will ever pull the same row.
DATA_ITERATOR = DF.itertuples(index=False) if not DF.empty else None


class HttpInjector(FastHttpUser):
    wait_time = constant(0)

    @task
    def inject(self):
        if DATA_ITERATOR is None:
            if self.environment.runner:
                self.environment.runner.quit()
            raise StopUser()

        try:
            row = next(DATA_ITERATOR)
            payload = {
                key: getattr(row, key)
                for key in PAYLOAD_KEYS if hasattr(row, key)
            }
        except StopIteration:
            if self.environment.runner:
                self.environment.runner.quit()
            raise StopUser()

        # Route the request based on the dynamic NUM_WORKERS
        merchant_id = payload['merchant_id']
        stable_hash = int(
            hashlib.md5(str(merchant_id).encode()).hexdigest(), 16)

        # Calculate port: 8001, 8002, 8003, etc...
        target_port = 8001 + (stable_hash % NUM_WORKERS)
        dynamic_url = f"http://10.0.2.7:{target_port}{ENDPOINT}"

        with self.client.post(dynamic_url, json=payload,
                              catch_response=True) as response:
            if response.status_code != 200:
                response.failure(f"Failed: {response.status_code}")
