IMAGE_NAME="fake-news-prediction-service"
IMAGE_TAG="v1"
DOCKER_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

test:
	pytest tests/

quality_checks:
	black .
	pylint --recursive=y .

build: quality_checks test
	docker build -f 02_deployment/Dockerfile -t ${DOCKER_IMAGE_NAME} .

integration_test: build
	bash ./integration_test/run.sh

publish: build integration_test
	LOCAL_IMAGE_NAME=${LOCAL_IMAGE_NAME} bash scripts/publish.sh

setup:
	pipenv install --dev
	pre-commit install
