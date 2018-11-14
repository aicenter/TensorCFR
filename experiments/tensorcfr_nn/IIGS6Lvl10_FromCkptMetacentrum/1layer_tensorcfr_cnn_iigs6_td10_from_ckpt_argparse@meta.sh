#!/usr/bin/env bash
#PBS -N CFR_NN_1layer_IIGS6Lvl10_ckpt
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb:cluster=^grimbold

# README
# This script runs CFR + CNN_IIGS6Lvl10 recorded from a checkpoint file.

# configure variables
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
EXPERIMENT_NAME="CFR_NN_1layer_IIGS6Lvl10_ckpt_experiments"
FRONTNODE_LOGS="${REPO_DIR}/logs/${EXPERIMENT_NAME}"
OUTFILE=${EXPERIMENT_NAME}_$(date -d "today" +"%Y%m%d%H%M").out

BASENAME=tensorcfr_CNN_IIGS6_td10_from_ckpt
SCRIPT=src.algorithms.tensorcfr_nn.${BASENAME}
COMMON_ARGS=""
CKPT_COMMON_DIR="${REPO_DIR}/src/algorithms/tensorcfr_nn/checkpoints"

PYTHON=python3
mkdir -p ${FRONTNODE_LOGS}
module add tensorflow-1.7.1-gpu-python3

cd ${REPO_DIR} || exit 1

# 1 layer
CKPT_DIR="${CKPT_COMMON_DIR}/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021836-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,r=C-138,t=1,tr=0.8"
CKPT_BASENAME="final_2018-11-11_07:46:15.ckpt"
CKPTS="--ckpt_dir ${CKPT_DIR} --ckpt_basename ${CKPT_BASENAME}"
ARCH="--extractor C-46 --regressor C-138"
CMD="${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} ${CKPTS}"

echo ${CMD} |& tee ${FRONTNODE_LOGS}/${OUTFILE}.1_layers
${CMD} |& tee ${FRONTNODE_LOGS}/${OUTFILE}.1_layers

# TODO remove this part left from scripts for @aic-ml
# # 5 layers
# CKPT_DIR="Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_030427-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8"
# CKPT_BASENAME="final_2018-11-11_16:52:00.ckpt"
# CKPTS="--ckpt_dir ${CKPT_COMMON_DIR} --ckpt_basename ${CKPT_BASENAME}"
# ARCH="--extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138"
# CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} ${CKPTS} |& tee ${OUTFILE}.5_layers
#
# ## 7-layers
# CKPT_DIR="src/algorithms/tensorcfr_nn/checkpoints/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_133801-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8"
# CKPT_BASENAME="final_2018-11-12_09:37:52.ckpt"
# CKPTS="--ckpt_dir ${CKPT_COMMON_DIR} --ckpt_basename ${CKPT_BASENAME}"
# ARCH="--extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138"
# CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} ${CKPTS} |& tee ${OUTFILE}.7_layers
