#!/usr/bin/env bash

PYTHON=python3.6
DATADIR="data/IIGS6Lvl10/IIGS6_1_6_false_true_lvl10_npz_900_seeds"  # Warning: copy the dataset into this location !!
COMMON_ARGS="--dataset_directory ${DATADIR} --epochs 100"

CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.algorithms.tensorcfr_nn.tensorcfr_CNN_IIGS6_td10_online_training ${COMMON_ARGS}
