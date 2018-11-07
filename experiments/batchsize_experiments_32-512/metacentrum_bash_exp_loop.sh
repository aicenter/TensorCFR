#!/usr/bin/env bash
for i in {32,64,128,256,512}
do
	qsub metacentrum_exp_1_bs${i}.sh
done