# Use Python 3.10 slim as the base image
FROM python:3.10-bullseye
RUN apt-get install -y git

# Upgrade pip
RUN pip install --upgrade pip

# Set the working directory to /app
WORKDIR /app
VOLUME psycop-common

# Dev experience
## Lazygit
RUN LAZYGIT_VERSION=$(curl -s "https://api.github.com/repos/jesseduffield/lazygit/releases/latest" | grep -Po '"tag_name": "v\K[^"]*') && curl -Lo lazygit.tar.gz "https://github.com/jesseduffield/lazygit/releases/latest/download/lazygit_${LAZYGIT_VERSION}_Linux_x86_64.tar.gz" && tar xf lazygit.tar.gz lazygit && install lazygit /usr/local/bin && rm -rf lazygit lazygit.tar.gzRUN curl https://raw.githubusercontent.com/jesseduffield/lazydocker/master/scripts/install_update_linux.sh | bash

COPY dev-requirements.txt ./
RUN pip install -r dev-requirements.txt

## Initialise pre-commit
### Remove the .git repo afterwards, since the dev container bind-mounts the development repo
COPY .pre-commit-config.yaml ./
RUN git init . && pre-commit run

# App install
## Each on their own line to have them be their own layers, optimising caching
COPY gpu-requirements.txt .
RUN pip install -r gpu-requirements.txt

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY test-requirements.txt .
RUN pip install -r test-requirements.txt

COPY . /app
RUN pip install -e .