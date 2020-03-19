FROM ubuntu:latest
#FROM mwirtz/python3.5-healpy:latest
LABEL com.example.label-with-value="model" version="1.0"
COPY requirements.txt /tmp/
RUN apt-get update && apt-get upgrade -y
RUN apt-get install python3.5
RUN echo "y" | apt-get install python3-pip
RUN pip3 install --requirement /tmp/requirements.txt
COPY . /tmp/DetectDiseaseTHS
