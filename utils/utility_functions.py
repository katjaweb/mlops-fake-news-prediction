"""
contains helper functions
"""

import io
import os
from datetime import datetime

import yaml
import boto3
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

# import time
# import mlflow
# import lightgbm as lgb
# from xgboost import XGBClassifier
# from sklearn.svm import LinearSVC
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import (recall_score, accuracy_score, precision_score)


def remove_outliers(features, target, num_cols, threshold=3, iterations=1):
    """
    Entfernt Ausreißer aus einem DataFrame anhand der Median Absolute Deviation (MAD)

    Parameter:
    df : pd.DataFrame - Der Eingabe-DataFrame.
    threshold : float - Der Schwellenwert für MAD (Standard: 3).
    iterations : int - Anzahl der Iterationen, um Ausreißer schrittweise zu entfernen (Standard: 1).

    Rückgabe:
    pd.DataFrame - DataFrame ohne Ausreißer.
    """
    features_sampled = features.copy()

    print((features_sampled.index != target.index).sum())

    # remove outliers with median absolute deviation
    for _ in range(iterations):
        median = features_sampled[num_cols].median()
        mad = (features_sampled[num_cols] - median).abs().median()

        mad_distance = (features_sampled[num_cols] - median).abs() / mad
        features_sampled = features_sampled[(mad_distance < threshold).all(axis=1)]

    # remove texts with less than 6 words
    features_sampled = features_sampled[features_sampled['text_word_count'] > 5]

    # remove non english texts
    features_sampled = features_sampled[features_sampled['language'] == 'en']

    target = target[features_sampled.index]

    return features_sampled, target


def test_feature_sets(model, feature_sets, y_train, validation_sets, y_test):
    """
    Function to evaluate models performance. Prints the classification report
    """
    for name, feature_set in feature_sets.items():
        mod = model
        mod.fit(feature_set, y_train)
        y_pred = model.predict(validation_sets[name])
        print(f'classification report for feature set: {name}')
        print(
            classification_report(
                y_test,
                y_pred,
                digits=3,
                target_names=['Real', 'Fake'],
            )
        )


def evaluate(y_test, y_pred):
    """
    Prints a classification report and displays a confusion matrix heatmap
    for the given true and predicted labels.

    Parameters:
        y_test (array-like): True labels.
        y_pred (array-like): Predicted labels.
    """
    print(
        classification_report(y_test, y_pred, digits=3, target_names=['Real', 'Fake'])
    )
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d')


def load_file_s3(s3_bucket, dataset_path, file_type):
    """
    Loads a CSV or Parquet file from an S3 bucket into a pandas DataFrame.

    Parameters:
        s3_bucket (str): Name of the S3 bucket.
        dataset_path (str): Path to the file within the S3 bucket.
        file_type (str): File format, either 'csv' or 'parquet'.

    Returns:
        pd.DataFrame: The loaded dataset.
    """
    s3 = boto3.client('s3')
    buffer = io.BytesIO()
    s3.download_fileobj(s3_bucket, dataset_path, buffer)
    buffer.seek(0)

    if file_type == 'csv':
        return pd.read_csv(buffer)

    return pd.read_parquet(buffer)


def upload_to_s3(file, s3_bucket, dataset):
    """
    Uploads a pandas Series or DataFrame to an S3 bucket as a CSV or Parquet file,
    including the current date in the filename.

    Parameters:
        file (pd.Series or pd.DataFrame): The data to upload.
        s3_bucket (str): Name of the target S3 bucket.
        dataset (str): Base name for the file (used in the filename).
    """
    s3_client = boto3.client('s3')
    config = load_config()

    # actual date
    date = datetime.now().strftime("%Y-%m-%d")

    if isinstance(file, pd.Series):
        buffer = io.BytesIO()
        file.to_csv(buffer, index=False)
        buffer.seek(0)
        # create file name with date
        file_key = f'datasets/{dataset}_{date}.csv'
        print(file_key)
    else:
        buffer = io.BytesIO()
        file.to_parquet(buffer, index=False)
        buffer.seek(0)
        # create file name with date
        file_key = f'datasets/{dataset}_{date}.parquet'
        print(file_key)

    s3_client.put_object(Bucket=s3_bucket, Key=file_key, Body=buffer.getvalue())
    print(f"File saved under {file_key} in {s3_bucket}.")
    config['data'][dataset] = file_key


def load_config():
    """
    load config file
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, '..', 'config', 'vars.yaml')
    with open(config_path, "r", encoding='utf-8') as file:
        config = yaml.safe_load(file)
    return config


# def log_model(
#     X_train, y_train, X_test, y_test, params, model_name, name_train_data, name_val_data
# ):  # pylint: disable=invalid-name
#     """
#     Trains and logs a classification model and its performance metrics to MLflow.

#     The function supports multiple model types (RandomForest, XGBoost, LinearSVC, LightGBM),
#     logs hyperparameters, training/inference time, performance metrics (accuracy, precision, recall),
#     and registers the model in the MLflow Model Registry.

#     Parameters:
#         X_train (pd.DataFrame): Training feature data.
#         y_train (pd.Series or np.array): Training labels.
#         X_test (pd.DataFrame): Validation feature data.
#         y_test (pd.Series or np.array): Validation labels.
#         params (dict): Model hyperparameters.
#         model_name (str): Name of the model to train and log.
#         name_train_data (str): Identifier for the training dataset (for logging).
#         name_val_data (str): Identifier for the validation dataset (for logging).
#     """
#     with mlflow.start_run():
#         mlflow.set_tag('model', model_name)
#         mlflow.log_params(params)
#         mlflow.log_param('train-data', name_train_data)
#         mlflow.log_param('valid-data', name_val_data)
#         mlflow.log_param('model_name', model_name)

#         # Modell auswählen
#         if model_name == 'RandomForest':
#             model = RandomForestClassifier(**params)
#         elif model_name == 'XGBoost':
#             model = XGBClassifier(**params)
#         elif model_name == 'LinearSVC':
#             model = LinearSVC(**params)
#         elif model_name == 'LightGBM':
#             model = lgb.LGBMClassifier(**params, objective='binary')
#         else:
#             raise ValueError(f'Unknown model: {model_name}')

#         # measure training time
#         start_time = time.time()

#         model.fit(X_train, y_train)

#         end_time = time.time()
#         training_time = end_time - start_time
#         mlflow.log_metric('training_time', training_time)

#         # Prediction for training and validation-set
#         y_train_pred = model.predict(X_train)

#         # measure inference time
#         start_inference_time = time.time()

#         y_test_pred = model.predict(X_test)

#         end_inference_time = time.time()
#         inference_time = end_inference_time - start_inference_time
#         mlflow.log_metric('inference_time', inference_time)

#         # Metrics for training set
#         train_accuracy = accuracy_score(y_train, y_train_pred)
#         train_precision = precision_score(y_train, y_train_pred)
#         train_recall = recall_score(y_train, y_train_pred)

#         # Metrics for test set
#         val_accuracy = accuracy_score(y_test, y_test_pred)
#         val_precision = precision_score(y_test, y_test_pred)
#         val_recall = recall_score(y_test, y_test_pred)

#         mlflow.log_metric('train_accuracy', train_accuracy)
#         mlflow.log_metric('train_precision', train_precision)
#         mlflow.log_metric('train_recall', train_recall)

#         mlflow.log_metric('accuracy', val_accuracy)
#         mlflow.log_metric('precision', val_precision)
#         mlflow.log_metric('recall', val_recall)

#         if model_name == 'XGBoost':
#             mlflow.xgboost.log_model(
#                 model,
#                 artifact_path='models',
#                 registered_model_name=model_name,
#             )
#         elif model_name == 'LightGBM':
#             mlflow.lightgbm.log_model(
#                 lgb_model=model,
#                 artifact_path='models',
#                 registered_model_name=model_name,
#             )
#         else:
#             mlflow.sklearn.log_model(
#                 sk_model=model,
#                 artifact_path='models',
#                 registered_model_name=model_name,
#             )
