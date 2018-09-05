#!/usr/bin/env bash

find src/ -name *.py | grep -v  __init__ | sed -e 's/\//./g' | rev | cut -d '.' -f 2- | rev \
    | xargs -I '{}' -t sh -c 'time -f "Duration: %e seconds" python3 -m {} >/dev/null || exit 255'
