FROM python:3.12.2-slim

RUN pip install -U pip
RUN pip install pipenv

WORKDIR /app

COPY ./02_deployment/Pipfile /app/Pipfile
COPY ./02_deployment/Pipfile.lock /app/Pipfile.lock

RUN pipenv install --system --deploy

RUN python -m spacy download en_core_web_sm

RUN apt-get update && apt-get install -y libgomp1

COPY ./utils/preprocessing.py /app/utils/preprocessing.py
COPY ./02_deployment/predict.py /app/predict.py

RUN mkdir -p /config
COPY ./config/vars.yaml /config/vars.yaml


EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app", "--log-level=debug", "--capture-output" ]


#Image build command:
# cd ../
# docker build -f 02_deployment/Dockerfile -t fake-news-prediction-service:v1 .

#Container running command:
# docker run -it --rm -p 9696:9696 -v /home/codespace/.aws:/root/.aws fake-news-prediction-service:v1

# Run test
# python 02_deployment/test.py
