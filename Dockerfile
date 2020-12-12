# For more information, please refer to https://aka.ms/vscode-docker-python
FROM eaglelanding_requirements:latest
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
ADD ./gym-jsbsim-simple /app/gym-jsbsim-simple
ADD ./landing-the-eagle /app/landing-the-eagle

# requirements werden schon in eaglelanding_requirements:latest installiert
#RUN python3 -m pip install -r ./RL-wrapper-gym/requirements.txt
#RUN python3 -m pip install -r ./gym-jsbsim-simple/requirements.txt
#RUN python3 -m pip install -r ./landing-the-eagle/requirements.txt


RUN python3 -m pip install -e ./RL-wrapper-gym
RUN python3 -m pip install -e ./gym-jsbsim-simple

# Switching to a non-root user, please refer to https://aka.ms/vscode-docker-python-user-rights
RUN useradd appuser && chown -R appuser /app
USER appuser

# During debugging, this entry point will be overridden. For more information, please refer to https://aka.ms/vscode-docker-python-debug
WORKDIR /app/landing-the-eagle

CMD ["python3", "learn_to_fly_test.py"]
