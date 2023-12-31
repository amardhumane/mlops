FROM python:3.7-slim

COPY requirements.txt .

RUN apt update
RUN apt install git -y
RUN pip install --upgrade pip --no-cache-dir
RUN pip install -r requirements.txt --no-cache-dir
RUN apt install graphviz -y
RUN pip install mlflow
