"""
Run experiment tracking with MLflow, find the best model and register
"""
# pylint: disable=redefined-outer-name

import os
import sys
import time
import warnings
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
# from utils import utilityFunctions
from utils.utility_functions import load_file_s3

import numpy as np
import mlflow
from mlflow.tracking import MlflowClient

from hyperopt import STATUS_OK, Trials, fmin, hp, tpe
from hyperopt.pyll.base import scope

import lightgbm as lgb
from xgboost import XGBClassifier

from sklearn.feature_extraction.text import CountVectorizer
# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

warnings.filterwarnings('ignore')

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "mlflow-fake-news-test-3"

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id


def run_optimization(X_train, y_train, X_test, y_test, model_name, search_space, num_trials, name_train_data, name_test_data): # pylint: disable=invalid-name
    """
    Performs hyperparameter optimization for a given classification model and logs the results using MLflow.

    The function uses `hyperopt` for tuning and supports various classification models such as 
    RandomForest, XGBoost, LinearSVC, and LightGBM. It builds a pipeline with Bag-of-Words vectorization,
    trains the model, computes evaluation metrics (accuracy, precision, recall, F1), measures training and 
    inference time, and logs all results to MLflow.

    Parameters:
        X_train (pd.DataFrame): Training feature data.
        y_train (pd.Series or np.array): Training target labels.
        X_test (pd.DataFrame): Test feature data.
        y_test (pd.Series or np.array): Test target labels.
        model_name (str): Name of the model ('RandomForest', 'XGBoost', 'LinearSVC', 'LightGBM').
        search_space (dict): Hyperparameter search space for the model.
        num_trials (int): Number of optimization iterations.
        name_train_data (str): Identifier for the training dataset (for logging purposes).
        name_test_data (str): Identifier for the test dataset (for logging purposes).
    """
    def objective(params):

        with mlflow.start_run():
            mlflow.set_tag("developer","katja")
            mlflow.set_tag("train-data", name_train_data)
            mlflow.set_tag("valid-data", name_test_data)
            mlflow.set_tag("model_name", model_name)
            mlflow.log_params(params)

            # Modell auswählen
            if model_name == "RandomForest":
                model = RandomForestClassifier(**params)
            elif model_name == "XGBoost":
                model = XGBClassifier(**params)
            elif model_name == "LinearSVC":
                model = LinearSVC(**params)
            elif model_name == "LightGBM":
                model = lgb.LGBMClassifier(**params, objective='binary')
            else:
                raise ValueError(f"Unknown model: {model_name}")

            text_col = 'title_text_clean'

            preprocessor_bow = ColumnTransformer([
            ('text', CountVectorizer(), text_col)
            ], remainder='passthrough')

            pipeline = Pipeline([
            ("preprocessor", preprocessor_bow),
            ("classifier", model)
            ])

            # measure training time
            start_time = time.time()

            pipeline.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time
            mlflow.log_metric("training_time", training_time)

            # Prediction for training and validation-set
            y_train_pred = pipeline.predict(X_train)

            # measure inference time
            start_inference_time = time.time()

            y_test_pred = pipeline.predict(X_test)

            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            mlflow.log_metric("inference_time", inference_time)

            # Metrics for training set
            train_accuracy = accuracy_score(y_train, y_train_pred)
            train_precision = precision_score(y_train, y_train_pred)
            train_recall = recall_score(y_train, y_train_pred)
            train_f1 = f1_score(y_train, y_train_pred)

            # Metrics for test set
            test_accuracy = accuracy_score(y_test, y_test_pred)
            test_precision = precision_score(y_test, y_test_pred)
            test_recall = recall_score(y_test, y_test_pred)
            test_f1 = f1_score(y_test, y_test_pred)

            mlflow.log_metric("train_accuracy", train_accuracy)
            mlflow.log_metric("train_precision", train_precision)
            mlflow.log_metric("train_recall", train_recall)
            mlflow.log_metric("train_f1", train_f1)

            mlflow.log_metric("accuracy", test_accuracy)
            mlflow.log_metric("precision", test_precision)
            mlflow.log_metric("recall", test_recall)
            mlflow.log_metric("f1", test_f1)

            mlflow.sklearn.log_model(
                sk_model=pipeline,
                artifact_path="models"
            )

        return {'loss': test_accuracy, 'status': STATUS_OK}

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate
    )


def get_current_production_metric(model_name, metric_name):
    """
    Retrieves the specified metric value from the current production version of a registered MLflow model.

    Parameters:
        model_name (str): The name of the registered MLflow model.
        metric_name (str): The name of the metric to retrieve.

    Returns:
        tuple: A tuple containing the metric value (float) and the model version, or (None, None) if unavailable.
    """
    try:
        # Get registered version by alias 'Production'
        version = client.get_model_version_by_alias(name=model_name, alias="Production")
        if not version:
            return None, None

        current_run_id = version.run_id
        current_metric = client.get_run(current_run_id).data.metrics.get(metric_name)

        return float(current_metric), current_version
    except Exception:
        return None, None


def register_model(current_metric, new_metric, model_name, best_run_id):
    """
    Registers a new model version in MLflow if it performs better than the current production model.

    The function compares the new model's performance metric to the current production model's metric.
    If the new model is better, it registers and promotes it to Production, archives the previous version,
    and updates the model description with the new metric value.

    Parameters:
        current_metric (float or None): Performance metric of the current production model.
        new_metric (float): Performance metric of the new candidate model.
        model_name (str): Name of the registered MLflow model.
        best_run_id (str): MLflow run ID of the best-performing new model.
    """
    client = MlflowClient()

    # Get current Production model
    try:
        current_prod_model = client.get_model_version_by_alias(model_name, "Production")
        current_version = current_prod_model.version
        current_metric = float(current_prod_model.description or -1)
    except Exception:
        current_prod_model = None
        current_version = None
        current_metric = None

    if current_metric is None or new_metric > current_metric:
        print("Registering new, better model...")

        # Register new model
        model_uri = f"runs:/{best_run_id}/model"
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        # Wait until model isREADY
        while client.get_model_version(name=model_name, version=result.version).status != "READY":
            time.sleep(1)

        # Archive old model (if exists)
        if current_prod_model:
            client.set_registered_model_alias(model_name, alias="Production", version=None)
            client.transition_model_version_stage(
                name=model_name,
                version=current_version,
                stage="Archived"
            )

        # Set new model to Production
        client.transition_model_version_stage(
            name=model_name,
            version=result.version,
            stage="Production"
        )
        client.set_registered_model_alias(model_name, alias="Production", version=result.version)

        # Save metric as description
        client.update_model_version(
            name=model_name,
            version=result.version,
            description=str(new_metric)
        )

        print(f"New model registered as Production: version {result.version}")

    else:
        print("New model is no better — skipping registration.")


# Load features and target

X_train = load_file_s3('fake-news-prediction', 'datasets/X_train_2025-04-08.parquet', 'parquet')
X_test = load_file_s3('fake-news-prediction', 'datasets/X_test_2025-04-08.parquet', 'parquet')
# X_val = load_file_s3('fake-news-prediction', 'datasets/X_val.parquet', 'parquet')

y_train = load_file_s3('fake-news-prediction', 'datasets/y_train_2025-04-08.csv', 'csv')
y_test = load_file_s3('fake-news-prediction', 'datasets/y_test_2025-04-08.csv', 'csv')
# y_val = load_file_s3('fake-news-prediction', 'datasets/y_val.csv', 'csv')

y_train = y_train.loc[:, 'label']
y_test = y_test.loc[:, 'label']
# y_val = y_val.loc[:, 'label']


# Experiment Tracking with MLflow

# model_names = ['RandomForest', 'XGBoost', 'LinearSVC', 'LightGBM']
model_names = ['LightGBM']

search_spaces = {
    # 'RandomForest': {
    #     'max_depth': scope.int(hp.quniform('max_depth', 50, 100, 1)),
    #     'n_estimators': scope.int(hp.quniform('n_estimators', 150, 250, 1)),
    #     'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
    #     'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
    #     'random_state': 42
    # },
    # 'XGBoost': {
    #     'max_depth': scope.int(hp.quniform("max_depth", 3, 12, 1)),
    #     'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 1)),
    #     'learning_rate': hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
    #     'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
    #     'reg_lambda' : hp.uniform('reg_lambda', 0,1),
    #     'seed': 0
    # },
    # 'LinearSVC': {
    #     'C': hp.uniform('C', 0, 10),
    #     "class_weight": hp.choice("class_weight", [None, "balanced"]),
    #     "loss": hp.choice("loss", ["hinge", "squared_hinge"]),
    #     "max_iter": scope.int(hp.quniform("max_iter", 3000, 5000, 100))
    # },
    'LightGBM': {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
        'learning_rate': hp.loguniform('learning_rate', -1, -0.5),
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 1)),
        'subsample': hp.uniform('subsample', 0.7, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.8, 1.0),
        'seed': 0
    }
}


for model_name in model_names:
    search_space = search_spaces[model_name]
    run_optimization(X_train=X_train,
                     y_train=y_train,
                     X_test=X_test,
                     y_test=y_test,
                     model_name=model_name,
                     search_space=search_space,
                     num_trials=2,
                     name_train_data='X_train_bow_num',
                     name_test_data='X_test_bow_num')


# Register best model

MODEL_NAME = "fake-news-model"
METRIC_NAME = "accuracy"

runs = client.search_runs(experiment_ids=[experiment_id], order_by=["metrics.accuracy DESC"])
best_run = runs[0]
best_run_id = best_run.info.run_id

# Get accuracy of best run
new_metric = best_run.data.metrics[METRIC_NAME]

current_metric, current_version = get_current_production_metric(MODEL_NAME, METRIC_NAME)

print(f"New run accuracy: {new_metric}")
print(f"Current model accuracy: {current_metric}")

register_model(current_metric, new_metric, MODEL_NAME, best_run_id)
