#!/usr/bin/env bash
size_of_generation=${1:-1}
num_generations=${2:-1000}
seed_offset=${3:-0}

last_generation=$((num_generations-1))
for generation in $(seq 0 $last_generation);
do
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
	seed=$((seed_offset + generation * size_of_generation))
	echo "Generation #$generation of size $size_of_generation, starting with dataset_seed $seed:"
	CUDA_VISIBLE_DEVICES=0 python3.6 -m src.nn.data.generate_data_of_IIGS6 --seed $seed --size $size_of_generation
	echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
done
