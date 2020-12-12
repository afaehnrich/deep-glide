# For more information, please refer to https://aka.ms/vscode-docker-python
FROM deep-glide/requirements:latest
#FROM debian:buster-slim


# Keeps Python from generating .pyc files in the container
ENV PYTHONDONTWRITEBYTECODE=1

# Turns off buffering for easier container logging
ENV PYTHONUNBUFFERED=1

# Install pip requirements

#RUN apt-get update && apt-get install -y python3-dev python3-pip

#RUN python --version
#RUN pip --version
WORKDIR /app
ADD ./RL-wrapper-gym /app/RL-wrapper-gym
ADD ./jsbgym-flex /app/jsbgym-flex
ADD ./deep-glide /app/deep-glide

# requirements werden schon in eaglelanding_requirements:latest installiert
#RUN python3 -m pip install -r ./RL-wrapper-gym/requirements.txt
#RUN python3 -m pip install -r ./jsbgym-flex/requirements.txt
#RUN python3 -m pip install -r ./deep-glide/requirements.txt


RUN python3 -m pip install -e ./RL-wrapper-gym
RUN python3 -m pip install -e ./jsbgym-flex

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
WORKDIR /app/deep-glide

CMD ["python3", "learn_to_fly_test.py"]
