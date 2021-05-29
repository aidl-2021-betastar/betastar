FROM python:3.9-slim

RUN apt-get update
RUN apt-get install build-essential git-core zlib1g-dev libjpeg-dev python-opengl xvfb -y

RUN set -ex; \
    apt-get install -y \
      bash \
      fluxbox \
      git \
      net-tools \
      novnc \
      supervisor \
      x11vnc \
      xterm \
      ffmpeg \
      unzip \
      libsdl2-dev libsdl2-image-dev libsdl2-mixer-dev libsdl2-ttf-dev libfreetype6-dev python3-setuptools python3-dev python3 libportmidi-dev

# from https://github.com/vwxyzjn/gym-pysc2
RUN wget -O ~/sc2.zip http://blzdistsc2-a.akamaihd.net/Linux/SC2.4.10.zip
RUN unzip -P iagreetotheeula ~/sc2.zip -d ~/
RUN rm -fr ~/sc2.zip

#RUN mv ~/StarCraftII/Libs/libstdc++.so.6 ~/StarCraftII/Libs/libstdc++.so.6.temp

WORKDIR /root

RUN wget -q https://github.com/deepmind/pysc2/releases/download/v1.2/mini_games.zip 

RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Melee.zip
RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season3.zip
RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season2.zip
RUN wget -q http://blzdistsc2-a.akamaihd.net/MapPacks/Ladder2017Season1.zip

# Uncompress zip files
RUN unzip -o mini_games.zip -d ~/StarCraftII/Maps/
RUN unzip -o -P iagreetotheeula Melee.zip -d ~/StarCraftII/Maps/
RUN unzip -o -P iagreetotheeula Ladder2017Season3.zip -d ~/StarCraftII/Maps/
RUN unzip -o -P iagreetotheeula Ladder2017Season2.zip -d ~/StarCraftII/Maps/
RUN unzip -o -P iagreetotheeula Ladder2017Season1.zip -d ~/StarCraftII/Maps/

# Delete zip files
RUN rm mini_games.zip
RUN rm Melee.zip
RUN rm Ladder2017Season3.zip
RUN rm Ladder2017Season2.zip
RUN rm Ladder2017Season1.zip


ENV HOME=/root \
    DEBIAN_FRONTEND=noninteractive \
    LANG=en_US.UTF-8 \
    LANGUAGE=en_US.UTF-8 \
    LC_ALL=C.UTF-8 \
    DISPLAY=:0.0 \
    DISPLAY_WIDTH=1024 \
    DISPLAY_HEIGHT=768 \
    RUN_XTERM=no \
    RUN_FLUXBOX=yes

ENV POETRY_VIRTUALENVS_CREATE=false

RUN pip install 'poetry==1.1.5'

ADD pyproject.toml .
ADD poetry.lock .
ADD betastar betastar
RUN poetry install --no-dev

ENTRYPOINT ["/bin/sh", "-c", "/usr/bin/xvfb-run -a $@", ""]
CMD ["betastar", "run"]
