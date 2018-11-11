#!/usr/bin/env bash

# @aic-ml: dataset of 17k TFRecord files
#PYTHON=python3.6
#DATADIR=/home/mathemage/beyond-deepstack/data/IIGS6/17450_datapoints_1_seed_per_file
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_TFRecords --dataset_directory ${DATADIR} --epochs 512 --extractor C-46 --regressor C-138
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_TFRecords --dataset_directory ${DATADIR} --epochs 512 --extractor C-46,C-46 --regressor C-138,C-138
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_TFRecords --dataset_directory ${DATADIR} --epochs 512 --extractor C-46,C-46,C-46 --regressor C-138,C-138,C-138
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_TFRecords --dataset_directory ${DATADIR} --epochs 512 --extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_TFRecords --dataset_directory ${DATADIR} --epochs 512 --extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138

ARCHS=(
"--extractor C-46 --regressor C-138"
"--extractor C-46,C-46 --regressor C-138,C-138"
"--extractor C-46,C-46,C-46 --regressor C-138,C-138,C-138"
"--extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138"
"--extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138"
)
for ARCH in "${ARCHS[@]}"
do
    NAME="CNN_${ARCH//[ -]/}_IIGS6Lvl10_17500seeds_tfrecords"
	CMD="qsub -N \"${NAME}\" -v \"ARCH=${ARCH}\" x_layers.sh "
	echo ${CMD}
	${CMD}
done
