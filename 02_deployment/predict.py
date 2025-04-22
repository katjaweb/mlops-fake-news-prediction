"""
Predicts whether news are fake or real, its probability and actual run id.
"""

# Test
import os
import sys

import yaml
import mlflow
import pandas as pd
from flask import Flask, jsonify, request

# pylint: disable=duplicate-code
here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils.preprocessing import prepare_features, apply_text_cleaner

current_dir = os.path.dirname(os.path.abspath(__file__))
config_path = os.path.join(current_dir, '..', 'config', 'vars.yaml')

with open(config_path, "r", encoding='utf-8') as file:
    config = yaml.safe_load(file)

production_run_id = config['mlflow']['production_run_id']
model_bucket = config['mlflow']['model_bucket']
experiment_id = config['mlflow']['experiment_id']

MODEL_LOCATION = (
    f's3://{model_bucket}/{experiment_id}/{production_run_id}/artifacts/models'
)
model = mlflow.sklearn.load_model(MODEL_LOCATION)


def feature_prep(news):
    """
    prepare_features creates new numerical features and applies NLP-Steps to the text
    """
    features = pd.DataFrame([news])
    features = prepare_features(features)
    features = apply_text_cleaner(features, column='title_text')
    return features


def predict(features):
    """
    Predicts whether the given input features correspond to fake or real news.

    Parameters:
        features (array-like): Input features for the model to make predictions.

    Returns:
        tuple: A label ('fake news' or 'real news') and the predicted probabilities for each class.
    """
    preds = model.predict(features)
    probas = model.predict_proba(features)
    label = ""
    if preds[0] == 1:
        label = 'fake news'
    elif preds[0] == 0:
        label = 'real news'
    return label, probas


app = Flask('fake-news-prediction')


@app.route('/predict', methods=['POST'])
def predict_endpoint():
    """
    API endpoint for predicting whether a given news article is fake or real.

    Expects a JSON payload via POST containing the news content. The input is
    preprocessed into features, passed to the prediction model, and the response
    includes the predicted label, probabilities for each class, and the model version.

    Returns:
        JSON response with prediction results or an error message on failure.
    """
    try:

        news = request.get_json()

        features = feature_prep(news)
        pred, proba = predict(features)

        result = {
            'label': pred,
            'probability being fake': round(float(proba[0][1]), 3),
            'probability being real': round(float(proba[0][0]), 3),
            'model_version': production_run_id,
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host='0.0.0.0', port=9696)
