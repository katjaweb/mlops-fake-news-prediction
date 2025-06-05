IMAGE_NAME="fake-news-prediction-service"
IMAGE_TAG="v1"
DOCKER_IMAGE_NAME="${IMAGE_NAME}:${IMAGE_TAG}"

test:
	pytest tests/

quality_checks:
	black .
	pylint --recursive=y .

build:
	docker build -f 02_deployment/Dockerfile -t ${DOCKER_IMAGE_NAME} .

integration_test: build
	bash ./integration_test/run.sh

run_process_data:
	python 01_development/process_data.py

start_mlflow:
	@bucket=$$(python3 -c "import yaml; print(yaml.safe_load(open('config/vars.yaml'))['mlflow']['model_bucket'])") && \
	mlflow server \
	  --backend-store-uri=sqlite:///mlflow.db \
	  --default-artifact-root=s3://$$bucket/


run_train:
	python 01_development/train.py

run_unit_tests:
	pipenv run pytest tests/

run_integration_test:
	bash -c "./integration_test/run.sh"

eb_deploy:
	bash -c "./02_deployment/deploy/deploy.sh"

# set up monitoring
monitoring:
	docker-compose up db adminer grafana -d --build
	sleep 10
	curl http://localhost:3000
	curl http://localhost:8080
	python 03_monitoring/monitoring.py

setup:
	pip install pipenv
	pipenv install --dev
	bash -c "chmod +x integration_test/run.sh"
	bash -c "chmod +x 02_deployment/deploy/deploy.sh"
	aws configure
