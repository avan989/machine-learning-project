FROM tensorflow/tensorflow:latest-py3
MAINTAINER Anh 

# copy project
RUN mkdir /app
WORKDIR /app
ADD . /app/

# copy entrypoint.sh
COPY ./entrypoint.sh /app/

# run entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
