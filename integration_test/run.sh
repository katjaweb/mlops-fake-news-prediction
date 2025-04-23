#!/bin/bash

# make bash file executable
# chmod +x integration_test/run.sh

# run integration_test
# ./integration_test/run.sh

if [[ -z "${GITHUB_ACTIONS}" ]]; then
  cd "$(dirname "$0")"
fi

SCRIPT_DIR=$(pwd)
PROJECT_ROOT=$(cd "$SCRIPT_DIR/.." && pwd)

IMAGE_NAME="fake-news-prediction-service"
IMAGE_TAG="v1"
DOCKER_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

# Build
echo "🔧 Building Docker image..."
# docker build -t ${IMAGE_NAME}:${IMAGE_TAG} ./02_deployment

docker build -f "${PROJECT_ROOT}/02_deployment/Dockerfile" -t "$DOCKER_IMAGE_NAME" "$PROJECT_ROOT"

# Run
echo "Starting container..."
docker run -it -d --rm \
  -p 9696:9696 \
  -v ~/.aws:/root/.aws \
  "$DOCKER_IMAGE_NAME"

sleep 5

CONTAINER_ID=$(docker ps -q -f "ancestor=$DOCKER_IMAGE_NAME")

if [ -z "$CONTAINER_ID" ]; then
    echo "No container is running for the image: $DOCKER_IMAGE_NAME"
    exit 1
fi

echo "Container with image $DOCKER_IMAGE_NAME is running (ID: $CONTAINER_ID)."

sleep 5

pipenv run python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} -eq 0 ]; then
    echo "Python script executed successfully."
else
    echo "Failed to execute Python script."
    docker logs $CONTAINER_ID
    exit ${ERROR_CODE}
fi

# Stop and remove the container
echo "Stopping container..."
docker stop "$CONTAINER_ID"
# docker rm ${CONTAINER_NAME}
echo "Container stopped.
