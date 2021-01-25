FROM pytorch/pytorch:latest
WORKDIR /predict_pytorch
RUN pip install --upgrade pip==19.3 \
&& pip install -U torch numpy joblib
COPY predict.py /predict_pytorch
ENTRYPOINT ["python", "predict.py"]