FROM python:3.7-alpine
MAINTAINER Anh 

# set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 0


# download dependency
RUN apk update \
	&& apk add gcc libxml2-dev libxslt-dev libc-dev

RUN pip install --upgrade pip
RUN pip install requests bs4 lxml
 
# copy project
RUN mkdir /app
WORKDIR /app
ADD . /app/

# copy entrypoint.sh
COPY ./entrypoint.sh /app/

# run entrypoint.sh
ENTRYPOINT ["/app/entrypoint.sh"]
