#!/usr/bin/env bash
for i in $(seq 7)
do
	qsub metacentrum_${i}_layers.sh
done
