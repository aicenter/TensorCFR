#!/usr/bin/env bash
for i in $(seq 4)
do
	qsub metacentrum_arch_exp_${i}.sh
done