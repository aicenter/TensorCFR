#!/usr/bin/env bash
for i in $(seq 3)
do
	qsub metacentrum_hyp_opt_exp_${i}.sh
done
