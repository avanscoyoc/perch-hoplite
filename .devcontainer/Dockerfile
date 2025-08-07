FROM python:3.11-slim

RUN apt-get update && apt-get install -y curl build-essential git gnupg curl lsb-release fuse ca-certificates \
    && rm -rf /var/lib/apt/lists/*

RUN curl -sSL https://install.python-poetry.org | python3 -

RUN export GCSFUSE_REPO=gcsfuse-lsb_release -c -s \
    && echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | tee /etc/apt/sources.list.d/gcsfuse.list \
    && curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - \
    && apt-get update && apt-get install -y gcsfuse
    
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /workspace

COPY . .

RUN poetry config virtualenvs.in-project true

RUN poetry install --no-interaction

CMD [ "bash" ]