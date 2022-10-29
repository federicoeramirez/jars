FROM python:3.8.15-slim

COPY jars/api /jars/api
COPY jars/ml_logic /jars/ml_logic
COPY jars/ml_logic /jars/ml_logic
COPY jars/data /jars/data
COPY requirements.txt /requirements.txt

RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD uvicorn jars.api.fast:app --host 0.0.0.0 --port 8000
