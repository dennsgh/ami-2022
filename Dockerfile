#syntax=docker/dockerfile:1

FROM tensorflow/tensorflow:latest

WORKDIR /server

ARG WORKINGDIR_ARG 

ENV FLASK_APP=flaskr \
    FLASK_ENV=production \
    WORKINGDIR=$WORKINGDIR_ARG \
    DATA=/mnt/smb/data \
    CONFIG=/mnt/smb/etc \
    MODEL=/mnt/smb/models \
    FRONTEND=$WORKINGDIR_ARG/frontend \
    PYTHONPATH=${PYTHONPATH}:./src:/server/src

# Install pipenv and compilation dependencies
RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y
COPY server/Pipfile server/Pipfile.lock ./
RUN pip install pipenv
RUN pipenv install --system

#install pytest in docker image
RUN pip install pytest

COPY . .

EXPOSE 5000
# comment out for debug
CMD ["/bin/bash","server/startup.sh"]
