FROM stablebaselines/rl-baselines3-zoo

# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1
# Fehler wegen falscher Codierung unterdrücken
ENV PYTHONIOENCODING = utf8

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

WORKDIR ${CODE_DIR}
# ADD ./requirements.txt ./requirements.txt
ADD . ${CODE_DIR}/deep-glide
RUN python3 -m pip install -r ${CODE_DIR}/deep-glide/requirements.txt
RUN git clone https://github.com/DLR-RM/rl-baselines3-zoo.git
# environment registration für die eigenen environemnts in rl-zoo integrieren
RUN cp -v -rt ${CODE_DIR}/rl-baselines3-zoo ${CODE_DIR}/deep-glide/rl_zoo/*
# deep-glide als Python-Modul installieren
RUN pip install -e ./deep-glide
WORKDIR ${CODE_DIR}/deep-glide
# Agent trainieren
CMD /bin/bash
