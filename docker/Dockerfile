FROM nvidia/cuda:11.0.3-cudnn8-devel-ubuntu18.04

ARG gid
ARG uid
ARG username

ENV DEBIAN_FRONTEND noninteractive

RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub

RUN apt-get update && apt-get install -y \
  git \
  wget \
  sudo \
  vim \
  python3-pip \
  python3-tk \
  python3-venv \
  python3-opencv

ENV UID=$uid
ENV GID=$gid
ENV USERNAME=$username
RUN mkdir -p /home/${USERNAME} && \
    echo "${USERNAME}:x:${UID}:${GID}:${USERNAME},,,:/${USERNAME}:/bin/bash" >> /etc/passwd && \
    echo "${USERNAME}:x:${GID}:" >> /etc/group && \
    echo "${USERNAME} ALL=(ALL) NOPASSWD: ALL" > /etc/sudoers.d/${USERNAME} && \
    chmod 0440 /etc/sudoers.d/${USERNAME} && \
    chown ${UID}:${GID} -R /home/${USERNAME} && \
    echo "export PATH=/home/${USERNAME}/.local/bin/:${PATH}" >> /home/${USERNAME}/.bash_profile && \
    echo "cd /home/${USERNAME}/btg" >> /home/${USERNAME}/.bash_profile

USER $username
ENV HOME /home/$username
WORKDIR /home/$username

RUN python3 -m pip install --upgrade pip
RUN pip3 install \
  ap_perf \
  torch==1.7.1+cu110 \
  torchvision==0.8.2+cu110 \
  torchaudio==0.7.2 -f https://download.pytorch.org/whl/torch_stable.html \
  pytorch-ignite==0.4.2 \
  tensorboard cython \
  tensorflow==2.4.1 \
  imbalanced-learn \
  pandas jupyterlab matplotlib seaborn
RUN pip3 install numpy --upgrade
RUN pip3 install torch-btg==0.0.1