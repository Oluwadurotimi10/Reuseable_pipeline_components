FROM python:3.7.1
WORKDIR /predict_data
RUN pip install --upgrade pip \
&& pip install -U tensorflow numpy joblib
COPY predict.py /predict_data
ENTRYPOINT ["python", "predict.py"]