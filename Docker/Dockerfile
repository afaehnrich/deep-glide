FROM nvidia/cuda:11.1-base-ubuntu20.04

# Hang bei Installation von tzdata verhindern (https://grigorkh.medium.com/fix-tzdata-hangs-docker-image-build-cdb52cc3360d)
ENV TZ=Europe/Berlin
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Fehler wegen falscher Codierung verhindern
ENV PYTHONIOENCODING = utf8

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

RUN apt-get -y update && apt-get -y install python3 python3-dev python3-pip git-all
RUN apt-get -y update \
    && apt-get -y install \
    ffmpeg \
    freeglut3-dev \
    swig \
    xvfb \
    libxrandr2 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*
    
ENV CODE_DIR /root/code
ENV VENV /root/venv

RUN mkdir -p ${CODE_DIR}/rl_zoo
RUN rm -rf $HOME/.cache/pip
    
RUN pip uninstall -y stable-baselines3 torch
RUN pip install torch==1.8.1+cu111 torchvision==0.9.1+cu111 torchaudio==0.8.1 -f https://download.pytorch.org/whl/torch_stable.html
RUN pip install gym==0.17.3
RUN pip install box2d-py==2.3.8 pybullet gym-minigrid scikit-optimize optuna pytablewriter seaborn pyyaml>=5.1 sb3-contrib>=1.1.0a7
RUN pip install wheel gym numpy matplotlib jsbsim toml datetime Flask tensorboard

ENV PATH=$VENV/bin:$PATH

#COPY docker/entrypoint.sh /tmp/
#RUN chmod +x /tmp/entrypoint.sh
#ENTRYPOINT ["/tmp/entrypoint.sh"]


WORKDIR ${CODE_DIR}
#JSBSim git clonen, damit wir Zugriff auf die Flugzeugmodelle haben
RUN git clone https://github.com/JSBSim-Team/jsbsim.git

# kompatible Versionen von stable-baselines3 und rl-baselines3-zoo von Git holen. Sonst gibts Fehlermeldung.

RUN git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
#RUN rm -r stable-baselines3
RUN git clone https://github.com/DLR-RM/stable-baselines3.git

ADD . ${CODE_DIR}/deep-glide
# environment registration für die eigenen environemnts in rl-zoo integrieren
RUN cp -v -rt ${CODE_DIR}/rl-baselines3-zoo ${CODE_DIR}/deep-glide/rl_zoo/*
# deep-glide als Python-Modul installieren
RUN pip install -e ./stable-baselines3
RUN pip install -e ./deep-glide
WORKDIR ${CODE_DIR}/deep-glide
# Agent trainieren
CMD /bin/bash
