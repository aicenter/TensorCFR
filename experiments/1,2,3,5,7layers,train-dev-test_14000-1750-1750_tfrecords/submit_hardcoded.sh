#!/usr/bin/env bash

layers=(1 2 3 5 7)
for i in "${layers[@]}"
do
	qsub ${i}_layers.sh
done
