#!/usr/bin/env bash

interpreter=bash
files=(
"./goofspiel/download_gambit_files.sh"
"./phantom_ttt/download_gambit_files.sh"
"./poker/download_gambit_files.sh"
)

for file in "${files[@]}"; do
    script=($file)
    command="$interpreter $script"
    echo ${command}
    ${command}
done
