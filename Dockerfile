# Use Python 3.10 slim as the base image
FROM python:3.10-bullseye
RUN apt-get install -y git

# Set the working directory to /app
WORKDIR /app
VOLUME psycop-common

# App install
## Each on their own line to have them be their own layers, optimising caching
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY gpu-requirements.txt .
RUN pip install -r gpu-requirements.txt

COPY test-requirements.txt .
RUN pip install -r test-requirements.txt

COPY dev-requirements.txt ./
RUN pip install -r dev-requirements.txt

COPY . /app
RUN pip install -e .