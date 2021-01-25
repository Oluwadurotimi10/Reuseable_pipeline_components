FROM python:3.7.1
WORKDIR /tf_data
RUN pip install --upgrade pip \
&& pip install -U tensorflow numpy 
COPY train.py /tf_data
ENTRYPOINT ["python", "train.py"]