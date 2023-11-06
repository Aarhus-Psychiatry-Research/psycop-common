FROM python:3.10

RUN apt-get update && apt-get install -y curl

# Install dev tools
COPY test-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r test-requirements.txt --no-compile

COPY dev-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r dev-requirements.txt --no-compile

COPY gpu-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r gpu-requirements.txt --no-compile

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/pip pip install -r requirements.txt --no-compile

# Set the working directory to /app
WORKDIR /app
VOLUME psycop-common

COPY . /app