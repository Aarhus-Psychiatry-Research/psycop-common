FROM python:3.10

ENV UV_SYSTEM_PYTHON=1

RUN apt-get update && apt-get install -y curl

# NVM and NPM are required for snyk
# Install nvm
# Explicitly set HOME environment variable 
ENV NVM_DIR=$HOME/.nvm
RUN mkdir -p $NVM_DIR
ENV NODE_VERSION=18.2.0

# Install nvm with node and npm
RUN curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.5/install.sh | bash \
    && . $NVM_DIR/nvm.sh \
    && nvm install $NODE_VERSION \
    && nvm alias default $NODE_VERSION \
    && nvm use default

ENV NODE_PATH=$NVM_DIR/v$NODE_VERSION/lib/node_modules
ENV PATH=$NVM_DIR/versions/node/v$NODE_VERSION/bin:$PATH

# Install snyk
RUN npm install -g snyk

# Install dev tools
# The cache mount caches downloaded packages for Docker
# The --no-compile options defers compilation to runtime, instead of install-time. This can dramatically save on build time, at the cost of slightly increased first-run times.
RUN pip install uv

COPY test-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r test-requirements.txt --no-compile

COPY dev-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r dev-requirements.txt --no-compile

COPY gpu-requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r gpu-requirements.txt --no-compile

COPY requirements.txt .
RUN --mount=type=cache,target=/root/.cache/uv uv pip install -r requirements.txt --no-compile

# Set the working directory to /app
WORKDIR /app
VOLUME psycop-common

COPY . /app
