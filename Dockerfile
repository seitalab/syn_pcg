FROM nvcr.io/nvidia/pytorch:24.01-py3

ARG USERNAME
ENV USERNAME=$USERNAME
ARG UID
ENV UID=$UID
ARG GID
ENV GID=$GID
ARG ORIGINAL_DATA_LOC
ENV ORIGINAL_DATA_LOC=$ORIGINAL_DATA_LOC

RUN apt-get update -y
RUN apt-get install -y make
RUN apt-get install -y lzma
RUN apt-get install -y liblzma-dev
RUN apt-get install -y gcc 
RUN apt-get install -y zlib1g-dev bzip2 libbz2-dev
RUN apt-get install -y libreadline8 
RUN apt-get install -y libreadline-dev
RUN apt-get install -y sqlite3 libsqlite3-dev
RUN apt-get install -y openssl libssl-dev build-essential 
RUN apt-get install -y git curl wget
RUN apt-get install -y vim
RUN apt-get install -y sudo
RUN apt-get install -y libffi-dev
RUN apt-get install -y libgl1-mesa-dev

RUN apt-get install -y lsb-release gnupg
RUN apt-get install -y python3.10 python3-pip
RUN apt-get install -y byobu

RUN rm /bin/sh && ln -s /bin/bash /bin/sh

RUN addgroup -gid $GID $USERNAME
RUN adduser $USERNAME --uid $UID --gid $GID
RUN usermod -aG sudo $USERNAME
RUN echo "$USERNAME:$USERNAME" | chpasswd

RUN mkdir -p $ORIGINAL_DATA_LOC
RUN chmod 777 $ORIGINAL_DATA_LOC
WORKDIR /home/$USERNAME
USER $USERNAME

ENV HOME /home/$USERNAME
ENV PYENV_ROOT $HOME/.pyenv
ENV PATH $PYENV_ROOT/bin:$PATH
ENV PATH $PYENV_ROOT/shims:$PYENV_ROOT/bin:$PATH

RUN git clone https://github.com/pyenv/pyenv.git ~/.pyenv
RUN echo 'eval "$(pyenv init -)"' >> ~/.bashrc
RUN source ~/.bashrc

RUN pyenv install 3.8.6
RUN pyenv global 3.8.6
RUN source ~/.bashrc

COPY requirements.txt /home/$USERNAME/
RUN pip install --upgrade pip
RUN pip install torch==2.1.1
RUN pip install scikit-learn==1.3.2 scipy==1.10.1 pandas==2.0.3 optuna==3.4.0 numpy==1.24.4 matplotlib==3.7.5
RUN pip install git+https://github.com/aaren/wavelets
RUN pip install PyYaml
RUN pip install torchvision==0.16.1
RUN pip install soundfile==0.12.1
RUN pip install timm==0.9.12
RUN pip install einops==0.8.0
RUN pip install opt-einsum==3.4.0
RUN pip install pytorch-lightning==2.4.0
RUN pip install optuna==3.4.0
RUN pip install timm==0.9.12