"""
Predicts whether news are fake or real, its probability and actual run id.
"""

import os
import sys
import mlflow
import pandas as pd

from flask import Flask, request, jsonify
from utils.preprocessing import prepare_features, apply_text_cleaner
sys.path.append(os.path.abspath('..'))

# from mlflow.tracking import MlflowClient

# tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
# mlflow.set_tracking_uri(tracking_uri)

# client = MlflowClient()
# model_name = "fake-news-model"
# version = client.get_model_version_by_alias(name=model_name, alias="Production")
# RUN_ID = version.run_id

# logged_model = f'runs:/{RUN_ID}/models'
# model = mlflow.pyfunc.load_model(logged_model)

model_bucket = os.getenv("MODEL_BUCKET", "fake-news-prediction")
experiment_id = os.getenv("MLFLOW_EXPERIMENT_ID", "4")
run_id = os.getenv("RUN_ID", "93fed103d82644b5b12f0805bc0e7547")

model_location = f"s3://{model_bucket}/{experiment_id}/{run_id}/artifacts/models"


model = mlflow.sklearn.load_model(model_location)

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
            'model_version': run_id
        }

        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host = '0.0.0.0', port=9696)
