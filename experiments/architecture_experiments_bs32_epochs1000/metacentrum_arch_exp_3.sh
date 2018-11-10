#!/bin/bash
#PBS -N ConvNet_IIGS6Lvl10_hyp_opt_1_npz
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script runs ConvNet_IIGS6Lvl10 with datasets of 900 seed files stored in NPZ for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  metacentrum.sh

# configure variables
FRONTNODE_HOME="/storage/brno2/home/seitzdom"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
EXPERIMENT_NAME="ConvNet_IIGS6Lvl10_900seeds_hyp_opt_arch"
FRONTNODE_LOGS="${REPO_DIR}/logs/${EXPERIMENT_NAME}"
OUTFILE=${EXPERIMENT_NAME}_$(date -d "today" +"%Y%m%d%H%M").out
FEATURES_DIM=46
BATCH_SIZE=32
EPOCHS=1000

mkdir -p ${FRONTNODE_LOGS}

module add tensorflow-1.7.1-gpu-python3

# run TensorCFR
cd ${REPO_DIR} || exit 1
PYTHON=python
DATASET_DIRECTORY="../data/IIGS6/IIGS6_1_6_false_true_lvl10_npz_900_seeds"
COMMON_ARGS="--dataset_directory ${DATASET_DIRECTORY} --batch_size $BATCH_SIZE --epochs $EPOCHS"
ARCH="--extractor C-$(($FEATURES_DIM*6)),C-$(($FEATURES_DIM*6)),C-$(($FEATURES_DIM*6)),C-$(($FEATURES_DIM*6)),C-$(($FEATURES_DIM*6)),C-$(($FEATURES_DIM*6))
--regressor C-$(($FEATURES_DIM*6)),C-$(($FEATURES_DIM*6))"
CMD="${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_NPZ ${COMMON_ARGS} ${ARCH}"

$CMD >${FRONTNODE_LOGS}/${OUTFILE}
