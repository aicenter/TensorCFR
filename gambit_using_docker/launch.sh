#!/usr/bin/env bash

docker run -ti --rm \
       -e DISPLAY=${DISPLAY} \
       -v /tmp/.X11-unix:/tmp/.X11-unix \
       -v ${HOME}:${HOME} \
       docker-gambit
