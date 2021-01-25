FROM pytorch/pytorch:latest
WORKDIR /dataloader
RUN pip install --upgrade pip==19.3 \
&& pip install -U torch numpy joblib
COPY define_dataloader.py /dataloader
ENTRYPOINT ["python", "define_dataloader.py"]