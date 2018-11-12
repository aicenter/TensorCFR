#!/usr/bin/env bash

reps=5
for i in `seq ${reps}`
do
	echo qsub 1_layers.sh -N "CNN_v${i}_1layer_14000-1750-1750_tfrecords"
done