"""
Pipeline for Monitoring
"""

import os
import sys
import time
import random
import logging
import datetime

import yaml
import numpy as np
import mlflow
import pandas as pd
import psycopg

# from prefect import flow, task
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metrics import (
    ColumnDriftMetric,
    DatasetDriftMetric,
    DatasetMissingValuesMetric,
)
from evidently.metric_preset import ClassificationPreset

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from utils import utility_functions as uf

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config', 'vars.yaml')
with open(config_path, "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

model_bucket = config['mlflow']['model_bucket']
experiment_id = config['mlflow']['experiment_id']
run_id = config['mlflow']['production_run_id']
path_reference_data = config['data']['X_train']
path_current_data = config['data']['X_val']
path_y_reference = config['data']['y_train']
path_y_current = config['data']['y_val']

# Configuration
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s]: %(message)s"
)
SEND_TIMEOUT = 10
DB_CONNECTION_STRING = (
    "host=localhost port=5432 dbname=test user=postgres password=example"
)
rand = random.Random()


# SQL command to create the table
CREATE_TABLE_STATEMENT = """
DROP TABLE IF EXISTS dummy_metrics;
CREATE TABLE metrics (
    timestamp TIMESTAMP,
    prediction_drift FLOAT,
    num_drifted_columns INTEGER,
    share_missing_values FLOAT,
    accuracy FLOAT,
    precision FLOAT,
    recall FLOAT
);
"""

# Load data and model
MODEL_LOCATION = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models'
model = mlflow.sklearn.load_model(MODEL_LOCATION)

reference_data = uf.load_file_s3(
    model_bucket, path_reference_data, 'parquet'
)
raw_data = uf.load_file_s3(
    model_bucket, path_current_data, 'parquet'
)

y_current = uf.load_file_s3(
    model_bucket, path_y_current, 'csv'
)
y_current = y_current.loc[:, 'label']
raw_data['actual'] = y_current

y_reference = uf.load_file_s3(
    model_bucket, path_y_reference, 'csv'
)
y_reference = y_reference.loc[:, 'label']
raw_data['actual'] = y_current

# Add date column to raw_data for testing
num_rows = len(raw_data)

start_date = pd.to_datetime("2025-03-01")
end_date = pd.to_datetime("2025-03-31")

random_timestamps = pd.to_datetime(
    np.random.uniform(start_date.value, end_date.value, size=num_rows)
).round('s')

raw_data["timestamp"] = random_timestamps

# days = 30  # Number of days for monitoring

# For each row, create a day (in modulo) and a random time
# raw_data["timestamp"] = [
#     start_date + pd.Timedelta(days=i % days, seconds=random.randint(0, 86399))
#     for i in range(num_rows)
# ]

raw_data = raw_data.sort_values("timestamp")

# Time period and column assignment
begin = datetime.datetime(2025, 3, 1, 0, 0)

num_cols = list(reference_data.select_dtypes(include='number').columns)
column_mapping = ColumnMapping(
    target='actual',
    prediction='prediction',
    numerical_features=num_cols,
    categorical_features=None,
)

# Evidently Report-configuration
report = Report(
    metrics=[
        ColumnDriftMetric(column_name='prediction'),
        DatasetDriftMetric(),
        DatasetMissingValuesMetric(),
        ClassificationPreset(),
    ]
)


# @task
def prep_db():
    """
    Initializes the PostgreSQL database.
    Creates the 'test' database and a table for storing metrics.
    """
    logging.info("Preparing database...")
    with psycopg.connect(
        "host=localhost port=5432 user=postgres password=example", autocommit=True
    ) as conn:
        res = conn.execute("SELECT 1 FROM pg_database WHERE datname='test'")
        if len(res.fetchall()) == 0:  # Checks whether the database 'test' exists.
            conn.execute(
                "create database test;"
            )  # Creates the database if it is missing.
        with psycopg.connect(
            "host=localhost port=5432 dbname=test user=postgres password=example"
        ) as conn:
            conn.execute(CREATE_TABLE_STATEMENT)  # Creates the table.


# @task
def calculate_metrics_postgresql(curr, i):
    """
    Calculates the metrics for a specific day and saves the results in the database.
    """
    logging.info('Processing entry no: %s ...', i)
    current_data = raw_data[
        (raw_data.timestamp >= (begin + datetime.timedelta(i)))
        & (raw_data.timestamp < (begin + datetime.timedelta(i + 1)))
    ]

    current_data = current_data.drop(columns=['timestamp'], axis=1)

    # Model predictions
    reference_preds = model.predict(reference_data)
    reference_data['prediction'] = reference_preds
    reference_data['actual'] = y_reference

    current_data = current_data.copy()
    current_data.fillna('', inplace=True)
    val_preds = model.predict(current_data)
    current_data['prediction'] = val_preds
    current_data['actual'] = y_current

    # Run Evidently Report
    report.run(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping,
    )
    result = report.as_dict()

    # Extract results
    prediction_drift = result["metrics"][0]["result"]["drift_score"]
    num_drifted_columns = result["metrics"][1]["result"]["number_of_drifted_columns"]
    share_missing_values = result["metrics"][2]["result"]["current"][
        "share_of_missing_values"
    ]
    # classification metrics for current data
    accuracy = result['metrics'][3]['result']['current']['accuracy']
    precision = result['metrics'][3]['result']['current']['precision']
    recall = result['metrics'][3]['result']['current']['recall']

    # Saves the metrics in the database.
    curr.execute(
        "insert into metrics(timestamp, prediction_drift, num_drifted_columns, share_missing_values, accuracy, precision, recall) values (%s, %s, %s, %s, %s, %s, %s)",
        (
            begin + datetime.timedelta(i),
            prediction_drift,
            num_drifted_columns,
            share_missing_values,
            accuracy,
            precision,
            recall,
        ),
    )


# @flow
def batch_monitoring_backfill():
    """
    Performs backfill monitoring and stores metrics for each day in the data area.
    """
    logging.info("Starting batch monitoring backfill...")
    prep_db()

    last_send = datetime.datetime.now() - datetime.timedelta(seconds=SEND_TIMEOUT)

    with psycopg.connect(
        "host=localhost port=5432 dbname=test user=postgres password=example",
        autocommit=True,
    ) as conn:
        for i in range(0, 30):  # Loop for each day in the defined period.
            with conn.cursor() as curr:  # Creates a cursor for database operations.
                calculate_metrics_postgresql(curr, i)  # Calculates and saves metrics.

            new_send = datetime.datetime.now()  # Current timestamp.
            seconds_elapsed = (
                new_send - last_send
            ).total_seconds()  # Calculate time difference.

            # Waits if the timeout was not observed.
            if seconds_elapsed < SEND_TIMEOUT:
                time.sleep(SEND_TIMEOUT - seconds_elapsed)

            # Updates the last timestamp.
            while last_send < new_send:
                last_send = last_send + datetime.timedelta(seconds=10)

            logging.info("Data sent.")


if __name__ == "__main__":
    batch_monitoring_backfill()
