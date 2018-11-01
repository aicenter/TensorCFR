#!/usr/bin/env bash
<<<<<<< HEAD
<<<<<<< HEAD
#for i in $(seq 7)
for i in $(seq 2)
=======
for i in $(seq 7)
>>>>>>> 209ae84... Copy `7layers,width500,batchsize512,train-dev-test_720-90-90`
=======
#for i in $(seq 7)
for i in $(seq 2)
>>>>>>> 4b4b87e...  Define `7layers,width500,batchsize512,train-dev-test_720-90-90` for 2 layers
do
	qsub metacentrum_${i}_layers.sh
done
