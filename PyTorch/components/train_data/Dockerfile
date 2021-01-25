FROM python:3.7.1
WORKDIR /train_data
RUN pip install -U torch numpy joblib
COPY train.py /train_data
ENTRYPOINT ["python", "train.py"]