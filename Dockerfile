FROM tensorflow/tensorflow:latest-py3
MAINTAINER Anh 

RUN pip install --upgrade pip tensorflow-hub
RUN pip3 install pandas keras nltk

# copy project
RUN mkdir /app
WORKDIR /app
ADD . /app/

# copy entrypoint.sh
COPY ./entrypoint.sh /app/

# run entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
