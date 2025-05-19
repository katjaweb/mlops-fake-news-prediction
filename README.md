# Fake News Detection Web-service

This project addresses the growing challenge of online disinformation by providing a web-based tool to detect fake news. Built with Flask, the service uses a supervised machine learning model (LightGBM) and natural language processing techniques to classify English news articles as real or fake. It also provides a probability score to indicate prediction confidence.

# Key features
* Text preprocessing with tokenization and stopword removal
* Feature extraction using Bag-of-Words (BoW)
* Model trained on a labeled dataset of English news

The goal is to support journalists, researchers, and the public in critically assessing the credibility of online information.

# About the Dataset

(WELFake) is a dataset of 72,134 news articles with 35,028 real and 37,106 fake news. For this, authors merged four popular news datasets (i.e. Kaggle, McIntire, Reuters, BuzzFeed Political).

The Dataset contains four columns: Serial number (starting from 0); Title (about the text news heading); Text (about the news content); and Label (0 = fake and 1 = real). There are 78098 data entries in csv file out of which only 72134 entries are accessed as per the data frame.

Published in:
IEEE Transactions on Computational Social Systems: pp. 1-13 (doi: 10.1109/TCSS.2021.3068519).

# Tech Stack

* **scikit-learn:** Core library for building and training the fake news classification model.
* **MLflow:** Used for tracking experiments, model registry, and reproducibility.
* **hyperopt:** Enables efficient hyperparameter optimization.
* **Amazon Web Service (AWS):** Used for data storage and model artifacts via S3.
* **Flask:** Lightweight web framework powering the backend API.
* **EvidentlyAI:** Monitors data and model performance to detect drift and quality issues.
* **PostgreSQL:** Stores metadata, logs, and model-related data.
* **Grafana:** Dashboard for monitoring system and model metrics.
* **Docker:** Containerizes the entire application for easy deployment and scalability.
* **Pytest, Pylint, black, GitHub pre-commit hooks:** Ensure code quality through unit testing, linting, and automatic code formatting.
* **Github Actions:** Automates continuous integration (CI) workflows for testing and quality checks.

# System architecture

**Text processing and feature pipeline**: A data pipeline that loads raw news data, cleans and enriches it by generating numerical and textual features, and prepares it for model training. It combines and analyzes text fields, extracts linguistic and statistical features, performs NLP-based text cleaning (lemmatization, tokenization, stopword and punctuation removal), and outputs a cleaned title_text_clean column. The processed dataset is then uploaded to an S3 bucket.

**Training pipeline:** This pipeline handles both initial training and retraining of classification models using preprocessed features from S3. It performs hyperparameter tuning with hyperopt, supports models like RandomForest, XGBoost, LinearSVC, and LightGBM, and uses a Bag-of-Words pipeline for text vectorization. This pipeline ensures reproducible, performance-driven model selection and versioning.

* Loads preprocessed data from S3
* Performs hyperparameter optimization
* Trains and evaluates multiple classification models
* Tracks all experiments, metrics, and artifacts with MLflow
* Measures and logs training and inference time
* Automatically registers the best-performing model
* In retraining mode, compares with the current production model and only registers a new one if it performs better (archiving the old model)

**Inference Pipeline:** A Flask-based web service that handles real-time inference for fake news detection. It processes incoming news text, loads a trained model from an S3 bucket, and returns a prediction label ("fake news" or "real news") along with class probabilities.

* Flask API endpoint for serving predictions via requests
* Preprocesses input text to match model-ready format
* Loads the latest registered model from S3
* Performs inference and returns classification results
* Outputs both predicted label and class probabilities

**Monitoring Pipeline:** A batch monitoring pipeline that tracks model and data quality over time using EvidentlyAI, stores computed metrics in a PostgreSQL database, and visualizes them with Grafana. It evaluates model performance and data drift daily using held-out labeled data, ensuring long-term model reliability.

* Batch Monitoring with EvidentlyAI for drift, missing values, and classification metrics
* Loads reference and validation data from S3 for comparison
* Predicts using the production model loaded from S3 (via MLflow)
* Calculates daily metrics like accuracy, precision, recall, drift score, and missing data share
* Stores results in PostgreSQL using a time-series structure
* Supports backfilling to retroactively monitor historical data
* Dashboards powered by Grafana for visualization and analysis
* Automatic table creation for metric storage if missing

**CI-Tests:** GitHub Actions workflow that runs on pushes and pull requests to the develop branch. It sets up a Python environment, installs dependencies, runs unit tests with Pytest, performs linting with Pylint, and conducts integration tests by deploying the Flask app and executing test scripts. AWS credentials are configured for secure access to required resources.

# Getting started

Make sure you have docker

Follow these steps to set up and run the project locally:

Clone the repository

<pre><code>git clone https://github.com/katjaweb/mlops-fake-news-prediction.git
cd mlops-fake-news-prediction</code></pre>

**2. Install dependencies**

Make sure you have Python 3.12+ and Pipenv installed.

```bash
make setup
```

The make setup command prepares the local development environment by installing all required dependencies, enabling code quality checks, configuring permissions for integration tests, and setting up AWS credentials. Specifically, it installs development packages using Pipenv, sets up Git pre-commit hooks for linting and formatting, makes the integration test script executable, and initializes AWS CLI configuration for accessing cloud resources.

**3. Download required NLP model**

```bash
pipenv run python -m spacy download en_core_web_sm
```

Run text processing and feature pipeline

```bash
make run_process_data
```

The `run_process_data` Makefile command executes the `process_data.py` script located in the `01_development` folder. This script handles raw data processing, including feature engineering and text cleaning, saves the cleaned data to S3, and prepares it for model training.

Run the training pipeline

```bash
make train
```

The `make train` command starts an MLflow tracking server with a local SQLite database as the backend and an S3 bucket as the artifact store. The command first launches the MLflow server and then executes the training script (`train.py`) to train a model and log its artifacts and metrics to the MLflow server.

**4. Run the application locally**

```bash
pipenv run python app.py
```

**5. Run tests**

```bash
pipenv run pytest tests/
```
