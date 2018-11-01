#!/usr/bin/env bash
<<<<<<< HEAD
#for i in $(seq 7)
for i in $(seq 2)
=======
for i in $(seq 7)
>>>>>>> 209ae84... Copy `7layers,width500,batchsize512,train-dev-test_720-90-90`
do
	qsub metacentrum_${i}_layers.sh
done
