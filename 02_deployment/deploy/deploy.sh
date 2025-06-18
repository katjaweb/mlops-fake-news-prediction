#!/bin/bash

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(realpath "$SCRIPT_DIR/../..")"
DEPLOY_DIR="$SCRIPT_DIR"
APP_NAME="fake-news-service"
ENV_NAME="fake-news-env"
REGION="eu-west-1"

cp "$PROJECT_ROOT/02_deployment/Pipfile" "$DEPLOY_DIR/"
cp "$PROJECT_ROOT/02_deployment/Pipfile.lock" "$DEPLOY_DIR/"
cp "$PROJECT_ROOT/02_deployment/predict.py" "$DEPLOY_DIR/"
mkdir -p "$DEPLOY_DIR/utils"
cp "$PROJECT_ROOT/utils/preprocessing.py" "$DEPLOY_DIR/utils/"
mkdir -p "$DEPLOY_DIR/config"
cp "$PROJECT_ROOT/config/vars.yaml" "$DEPLOY_DIR/config/"

echo "All files copied to $DEPLOY_DIR"

# Initialize EB (only required the first time)
if [ ! -d ".elasticbeanstalk" ]; then
  echo "Initialize Elastic Beanstalk Environment"
  eb init -p docker -r "$REGION" "$APP_NAME"
fi

# Create or update environment
if eb status "$ENV_NAME" > /dev/null 2>&1; then
  echo "Deploy new version to existing environment"
  eb deploy "$ENV_NAME"
else
  echo "Create new Elastic Beanstalk environment"
  eb create "$ENV_NAME"
fi

echo "Clean up deploy directory"
rm Pipfile
rm Pipfile.lock
rm predict.py
rm utils/preprocessing.py
rm config/vars.yaml
rmdir utils 2>/dev/null || true
rmdir config 2>/dev/null || true

echo "deployment completed."
