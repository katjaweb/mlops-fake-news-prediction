"""
Run experiment tracking with MLflow, find the best model and register
"""

# pylint: disable=redefined-outer-name

import os
import sys
import time
import warnings
from datetime import datetime

import yaml
import numpy as np
import mlflow
from mlflow.entities import ViewType
from mlflow.tracking import MlflowClient
import lightgbm as lgb
from xgboost import XGBClassifier
from hyperopt import STATUS_OK, Trials, hp, tpe, fmin
from hyperopt.pyll.base import scope
from sklearn.svm import LinearSVC
from sklearn.compose import ColumnTransformer

# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))
from utils.utility_functions import load_file_s3

warnings.filterwarnings('ignore')

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config', 'vars.yaml')
with open(config_path, "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = config['mlflow']['experiment_name']

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(EXPERIMENT_NAME)

client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
experiment_id = mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id
config['mlflow']['experiment_id'] = experiment_id
model_bucket = config['mlflow']['model_bucket']



def run_optimization(
    X_train,
    y_train,
    X_test,
    y_test,
    model_name,
    search_space,
    num_trials,
    name_train_data,
    name_test_data,
):  # pylint: disable=invalid-name
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
            mlflow.set_tag('developer', 'katja')
            mlflow.set_tag('train-data', name_train_data)
            mlflow.set_tag('valid-data', name_test_data)
            mlflow.set_tag('model_name', model_name)
            mlflow.log_params(params)

            # Select model
            if model_name == 'RandomForest':
                model = RandomForestClassifier(**params)
            elif model_name == 'XGBoost':
                model = XGBClassifier(**params)
            elif model_name == 'LinearSVC':
                model = LinearSVC(**params)
            elif model_name == 'LightGBM':
                model = lgb.LGBMClassifier(**params, objective='binary')
            else:
                raise ValueError(f'Unknown model: {model_name}')

            text_col = 'title_text_clean'

            preprocessor_bow = ColumnTransformer(
                [('text', CountVectorizer(), text_col)], remainder='passthrough'
            )

            pipeline = Pipeline(
                [('preprocessor', preprocessor_bow), ('classifier', model)]
            )

            # measure training time
            start_time = time.time()

            pipeline.fit(X_train, y_train)

            end_time = time.time()
            training_time = end_time - start_time
            mlflow.log_metric('training_time', training_time)

            # Prediction for training and validation-set
            y_train_pred = pipeline.predict(X_train)

            # measure inference time
            start_inference_time = time.time()

            y_test_pred = pipeline.predict(X_test)

            end_inference_time = time.time()
            inference_time = end_inference_time - start_inference_time
            mlflow.log_metric('inference_time', inference_time)

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

            mlflow.log_metric('train_accuracy', train_accuracy)
            mlflow.log_metric('train_precision', train_precision)
            mlflow.log_metric('train_recall', train_recall)
            mlflow.log_metric('train_f1', train_f1)

            mlflow.log_metric('accuracy', test_accuracy)
            mlflow.log_metric('precision', test_precision)
            mlflow.log_metric('recall', test_recall)
            mlflow.log_metric('f1', test_f1)

            input_example = X_train.iloc[[0]]

            mlflow.sklearn.log_model(sk_model=pipeline, artifact_path='models', input_example=input_example)

        return {'loss': test_accuracy, 'status': STATUS_OK}

    rstate = np.random.default_rng(42)
    fmin(
        fn=objective,
        space=search_space,
        algo=tpe.suggest,
        max_evals=num_trials,
        trials=Trials(),
        rstate=rstate,
    )


def evaluate_model_from_run(
    run_id, X_val, y_val
):  # pylint: disable=invalid-name
    """Loads a model from an MLflow run and evaluates it on the validation data."""
    # model_location = f's3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models'
    run = client.get_run(run_id)
    model_location = os.path.join(run.info.artifact_uri, "models")
    print(f"Evaluating model from run_id={run_id} at location={model_location}")
    model = mlflow.sklearn.load_model(model_location)
    preds = model.predict(X_val)
    return accuracy_score(y_val, preds)


def get_current_production_info(model_name):
    """
    Retrieves the current production run and version, if available.
    """
    try:
        version = client.get_model_version_by_alias(name=model_name, alias='Production')
        if not version:
            return None, None

        current_run_id = version.run_id
        current_version = version.version
        return current_run_id, current_version
    except Exception:
        return None, None


def register_model_if_better(
    config,
    config_path,
    model_name,
    new_run_id,
    X_val,
    y_val,
):  # pylint: disable=invalid-name
    """
    Evaluates the current production model and the new model on X_val/y_val
    and only registers the new model if it is better.
    """
    current_run_id, current_version = get_current_production_info(model_name)

    # Evaluate new model
    new_metric = evaluate_model_from_run(
        new_run_id, X_val, y_val
    )
    print(f'New model (run {new_run_id}), accuracy: {new_metric:.4f}')

    # Evaluate current model (if available)
    if current_run_id:
        current_metric = evaluate_model_from_run(
            current_run_id, X_val, y_val
        )
        print(
            f'Old production model (run {current_run_id}), accuracy: {current_metric:.4f}'
        )
    else:
        current_metric = None
        print('No production model found.')

    # Only deploy if better
    if current_metric is None or new_metric > current_metric:
        print('New model is better - will be registered...')

        model_uri = f'runs:/{new_run_id}/model'
        result = mlflow.register_model(model_uri=model_uri, name=model_name)

        config['mlflow']['production_run_id'] = new_run_id
        with open(config_path, 'w', encoding='utf-8') as file:
            yaml.safe_dump(config, file)
        print(f'Saved run-ID {new_run_id} in {config_path}.')

        # Wait until registered
        while (
            client.get_model_version(name=model_name, version=result.version).status
            != 'READY'
        ):
            time.sleep(1)

        # Archive old model (if available)
        if current_version:
            client.transition_model_version_stage(
                name=model_name, version=current_version, stage='Archived'
            )

        # Bringing new model into production
        client.transition_model_version_stage(
            name=model_name, version=result.version, stage='Production'
        )
        client.set_registered_model_alias(
            model_name, alias='Production', version=result.version
        )

        # Save metric in description
        client.update_model_version(
            name=model_name, version=result.version, description=str(new_metric)
        )

        print(f'Model version {result.version} has been put into production.')
    else:
        print('New model is no better - remains unregistered.')


# Load features and target

X_train = load_file_s3(model_bucket, config['data']['X_train'], 'parquet')
X_test = load_file_s3(model_bucket, config['data']['X_test'], 'parquet')
X_val = load_file_s3(model_bucket, config['data']['X_val'], 'parquet')

y_train = load_file_s3(model_bucket, config['data']['y_train'], 'csv')
y_test = load_file_s3(model_bucket, config['data']['y_test'], 'csv')
y_val = load_file_s3(model_bucket, config['data']['y_val'], 'csv')

print("Shapes after loading:")
print('train data:', X_train.shape, y_train.shape)
print('val data:', X_val.shape, y_val.shape)
print('test data:', X_test.shape, y_test.shape)

y_train = y_train.loc[:, 'label']
y_test = y_test.loc[:, 'label']
y_val = y_val.loc[:, 'label']


# Experiment Tracking with MLflow

model_names = ['RandomForest', 'XGBoost', 'LinearSVC', 'LightGBM']
# model_names = ['LightGBM']

search_spaces = {
    'RandomForest': {
        'max_depth': scope.int(hp.quniform('max_depth', 50, 100, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 150, 250, 1)),
        'min_samples_split': scope.int(hp.quniform('min_samples_split', 2, 20, 1)),
        'min_samples_leaf': scope.int(hp.quniform('min_samples_leaf', 1, 4, 1)),
        'random_state': 42
    },
    'XGBoost': {
        'max_depth': scope.int(hp.quniform("max_depth", 3, 12, 1)),
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 1)),
        'learning_rate': hp.loguniform("learning_rate", np.log(0.01), np.log(0.3)),
        'reg_alpha' : hp.quniform('reg_alpha', 40,180,1),
        'reg_lambda' : hp.uniform('reg_lambda', 0,1),
        'seed': 0
    },
    'LinearSVC': {
        'C': hp.uniform('C', 0, 10),
        "class_weight": hp.choice("class_weight", [None, "balanced"]),
        "loss": hp.choice("loss", ["hinge", "squared_hinge"]),
        "max_iter": scope.int(hp.quniform("max_iter", 3000, 5000, 100))
    },
    'LightGBM': {
        'max_depth': scope.int(hp.quniform('max_depth', 4, 15, 1)),
        'learning_rate': hp.loguniform('learning_rate', -1, -0.5),
        'n_estimators': scope.int(hp.quniform('n_estimators', 50, 150, 1)),
        'subsample': hp.uniform('subsample', 0.7, 1.0),
        'colsample_bytree': hp.uniform('colsample_bytree', 0.8, 1.0),
        'seed': 0,
    }
}


start_time = datetime.now()

for model_name in model_names:
    search_space = search_spaces[model_name]
    run_optimization(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        model_name=model_name,
        search_space=search_space,
        num_trials=2,
        name_train_data='X_train_bow_num',
        name_test_data='X_test_bow_num',
    )


# Register best model

MODEL_NAME = 'fake-news-model'

runs = client.search_runs(
    experiment_ids=[experiment_id],
    filter_string=f'start_time >= {int(start_time.timestamp() * 1000)}',
    run_view_type=ViewType.ACTIVE_ONLY,
    order_by=['metrics.accuracy DESC'],
)
best_new_run = runs[0]
best_new_run_id = best_new_run.info.run_id


register_model_if_better(
    config=config,
    config_path=config_path,
    model_name=MODEL_NAME,
    new_run_id=best_new_run_id,
    X_val=X_val,
    y_val=y_val,
)
