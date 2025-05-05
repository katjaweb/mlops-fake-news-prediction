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

run_process_data:
	python 01_development/process_data.py

start_mlflow:
	mlflow server \
	  --backend-store-uri=sqlite:///mlflow.db \
	  --default-artifact-root=s3://fake-news-prediction/ \
	  --host 0.0.0.0 --port 5000 &

run_train: start_mlflow
	python 01_development/train.py

# set up monitoring
monitoring:
	docker-compose up db adminer grafana -d --build
	sleep 5
	curl http://localhost:3000
	curl http://localhost:8080
	python 03_monitoring/monitoring.py

setup:
	pipenv install --dev
	pre-commit install
	bash -c "chmod +x integration/run.sh"
	aws configure
