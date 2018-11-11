#!/usr/bin/env bash

PYTHON=python3.6
BASENAME=tensorcfr_CNN_IIGS6_td10_online_training
SCRIPT=src.algorithms.tensorcfr_nn.${BASENAME}
COMMON_ARGS=""
OUTDIR=/home/mathemage/beyond-deepstack/TensorCFR/experiments/tensorcfr_nn/IIGS6Lvl10_FromCkpt
OUTFILE=${OUTDIR}/${BASENAME}.out

# CUDA_VISIBLE_DEVICES=1 python3.6 -m src.algorithms.tensorcfr_nn.tensorcfr_CNN_IIGS6_td10_from_ckpt --extractor C-46 --regressor C-138

# 1 layers
ARCH="--extractor C-46 --regressor C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} |& tee ${OUTFILE}.1_layers

# 2 layers
ARCH="--extractor C-46,C-46 --regressor C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} |& tee ${OUTFILE}.2_layers

# 3 layers
ARCH="--extractor C-46,C-46,C-46 --regressor C-138,C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} |& tee ${OUTFILE}.3_layers

# 5 layers
ARCH="--extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} |& tee ${OUTFILE}.5_layers

# 7-layers
ARCH="--extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} |& tee ${OUTFILE}.7_layers
