FROM codegram/sc2:4.10

ARG CUDA_VERSION=11.0.3
FROM nvidia/cuda:${CUDA_VERSION}-base-ubuntu20.04
ARG PYTHON_VERSION=3.9.4
ARG CUDA_VERSION

COPY --from=0 /root/StarCraftII /root/StarCraftII

# Install ubuntu packages
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        git \
        curl \
        ca-certificates \
        sudo \
        locales \
        openssh-server \
        vim && \
    # Remove the effect of `apt-get update`
    rm -rf /var/lib/apt/lists/* && \
    # Make the "en_US.UTF-8" locale
    localedef -i en_US -c -f UTF-8 -A /usr/share/locale/locale.alias en_US.UTF-8
ENV LANG en_US.utf8

# Setup timezone
ENV TZ=Europe/Madrid
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# Install miniconda (python)
# Referenced PyTorch's Dockerfile:
#   https://github.com/pytorch/pytorch/blob/master/docker/pytorch/Dockerfile
RUN curl -o miniconda.sh https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh && \
    chmod +x miniconda.sh && \
    ./miniconda.sh -b -p conda && \
    rm miniconda.sh && \
    conda/bin/conda install -y python=$PYTHON_VERSION && \
    conda/bin/conda install -c pytorch pytorch \
    conda/bin/conda clean -ya
ENV PATH $HOME/conda/bin:$PATH
RUN touch $HOME/.bashrc && \
    echo "export PATH=$HOME/conda/bin:$PATH" >> $HOME/.bashrc

ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install 'poetry==1.1.5'

ADD pyproject.toml .
ADD poetry.lock .
ADD betastar betastar
RUN poetry install --no-dev