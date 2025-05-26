"""
integration test for web-service and docker
"""

import os
import sys
import requests

here = os.path.dirname(__file__)
sys.path.append(os.path.join(here, '..'))

from utils import utility_functions as uf

config = uf.load_config()

SERVICE_URL = 'http://localhost:9696'
PREDICT_ENDPOINT = f'{SERVICE_URL}/predict'

dummy_data = {
    "title": "Breaking news: test event",
    "text": "This is a test to see if the prediction endpoint is functioning correctly.",
}


def test_service_running(dummy_news):
    """
    tests if the service is available
    """
    try:
        response = requests.post(PREDICT_ENDPOINT, json=dummy_news, timeout=5)
        assert (
            response.status_code == 200
        ), f"Service not available: {response.status_code}"
        print("Service is running.")
    except requests.exceptions.RequestException as e:
        raise AssertionError(f"Service check failed: {e}") from e


def test_model_loaded(dummy_news):
    """
    Test if the model is successfully loaded and outputs the right model version
    """
    response = requests.post(PREDICT_ENDPOINT, json=dummy_news, timeout=5)
    assert response.status_code == 200, f"Prediction failed: {response.status_code}"
    result = response.json()
    assert (
        "model_version" in result and result["model_version"]
    ), "Model version not found in response."
    assert result["model_version"] == config['mlflow']['production_run_id']
    print(f"Model loaded, version: {result['model_version']}")
    print('response:', response.json())


if __name__ == "__main__":
    test_service_running(dummy_news=dummy_data)
    test_model_loaded(dummy_news=dummy_data)
    print("All tests passed.")
