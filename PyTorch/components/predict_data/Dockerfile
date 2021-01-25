FROM python:3.7.1
WORKDIR /predict_data
RUN pip install -U torch numpy
COPY predict.py /predict_data
ENTRYPOINT ["python", "predict.py"]