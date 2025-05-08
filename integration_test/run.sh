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
echo 'Building Docker image...'
docker build -f "${PROJECT_ROOT}/02_deployment/Dockerfile" -t "$DOCKER_IMAGE_NAME" "$PROJECT_ROOT"

# Run
echo 'Starting container...'

if [ "$GITHUB_ACTIONS" = "true" ]; then
  echo "Running inside GitHub Actions"

  docker run -it -d --rm \
    -p 9696:9696 \
    -e AWS_ACCESS_KEY_ID \
    -e AWS_SECRET_ACCESS_KEY \
    -e AWS_REGION \
    "$DOCKER_IMAGE_NAME"

else
  echo "Running locally"

  docker run -it -d --rm \
    -p 9696:9696 \
    -v ~/.aws:/root/.aws \
    "$DOCKER_IMAGE_NAME"
fi

sleep 5

CONTAINER_ID=$(docker ps -q -f "ancestor=$DOCKER_IMAGE_NAME")

docker logs "$CONTAINER_ID"

if [ -z "$CONTAINER_ID" ]; then
    echo "No container is running for the image: $DOCKER_IMAGE_NAME"

    echo "Checking exited containers for logs..."
    EXITED_CONTAINER_ID=$(docker ps -a -q -f "ancestor=$DOCKER_IMAGE_NAME")

    if [ -n "$EXITED_CONTAINER_ID" ]; then
        echo 'Logs from exited container:'
        docker logs "$EXITED_CONTAINER_ID"
    else
        echo 'No exited container found for image.'
    fi

    exit 1
fi

echo "Container with image $DOCKER_IMAGE_NAME is running (ID: $CONTAINER_ID)."

echo 'Wait for the Flask service at localhost:9696 ...'

# Wait for flask for up to 10 seconds
for i in {1..10}; do
    if curl -s http://localhost:9696 >/dev/null; then
        echo "Flask is ready"
        break
    fi
    echo "Not ready yet... Try $i"
    sleep 1
done

# Check whether the server was reachable
if ! curl -s http://localhost:9696 >/dev/null; then
    echo 'Flask service did not respond in time.'
fi

echo 'Starting integration-test...'
pipenv run python integration_test.py

ERROR_CODE=$?

if [ ${ERROR_CODE} -eq 0 ]; then
    echo 'Python script executed successfully.'
else
    echo 'Failed to execute Python script.'
    docker logs $CONTAINER_ID
    exit ${ERROR_CODE}
fi

# Stop and remove the container
echo 'Stopping container...'
docker stop "$CONTAINER_ID"
# docker rm ${IMAGE_NAME}
echo 'Container stopped.'
