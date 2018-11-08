#!/usr/bin/env bash

PYTHON=python3.6
SCRIPT=src.algorithms.tensorcfr_nn.tensorcfr_CNN_IIGS6_td10_online_training
DATADIR="data/IIGS6Lvl10/IIGS6_1_6_false_true_lvl10_npz_900_seeds"  # Warning: copy the dataset into this location !!
COMMON_ARGS="--dataset_directory ${DATADIR} --epochs 1000"
OUTDIR=/home/mathemage/beyond-deepstack/TensorCFR/experiments/tensorcfr_nn/IIGS6Lvl10

# CUDA_VISIBLE_DEVICES=1 nohup python3.6 -m src.algorithms.tensorcfr_nn.tensorcfr_CNN_IIGS6_td10_online_training --dataset_directory data/IIGS6Lvl10/IIGS6_1_6_false_true_lvl10_npz_900_seeds --epochs 10 --extractor C-46 --regressor C-138 &>tensorcfr_CNN_IIGS6_td10_online_training.out.1_layer &

# 1 layers
ARCH="--extractor C-46 --regressor C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} &>${OUTDIR}/tensorcfr_CNN_IIGS6_td10_online_training.out.1_layers

# 2 layers
ARCH="--extractor C-46,C-46 --regressor C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} &>${OUTDIR}/tensorcfr_CNN_IIGS6_td10_online_training.out.2_layers

# 3 layers
ARCH="--extractor C-46,C-46,C-46 --regressor C-138,C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} &>${OUTDIR}/tensorcfr_CNN_IIGS6_td10_online_training.out.3_layers

# 5 layers
ARCH="--extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} &>${OUTDIR}/tensorcfr_CNN_IIGS6_td10_online_training.out.5_layers

# 7-layers
ARCH="--extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} &>${OUTDIR}/tensorcfr_CNN_IIGS6_td10_online_training.out.7_layers
