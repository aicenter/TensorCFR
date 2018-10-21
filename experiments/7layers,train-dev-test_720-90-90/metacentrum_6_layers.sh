#!/bin/bash
#PBS -N ConvNet_IIGS6Lvl10_6xlayers_900seeds_npz
#PBS -q gpu
#PBS -l walltime=10:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script runs ConvNet_IIGS6Lvl10 with datasets of 900 seed files stored in NPZ for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  metacentrum.sh

# configure variables
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
EXPERIMENT_NAME="ConvNet_IIGS6Lvl10_900seeds_npz_experiments"
FRONTNODE_LOGS="${REPO_DIR}/logs/${EXPERIMENT_NAME}"
OUTFILE=${EXPERIMENT_NAME}_$(date -d "today" +"%Y%m%d%H%M").out

mkdir -p ${FRONTNODE_LOGS}

module add tensorflow-1.7.1-gpu-python3

# run TensorCFR
cd ${REPO_DIR} || exit 1
PYTHON=python
DATASET_DIRECTORY="../../../data/IIGS6/IIGS6_1_6_false_true_lvl10_npz_900_seeds"
COMMON_ARGS="--dataset_directory ${DATASET_DIRECTORY} --epochs 50000"
ARCH="--extractor C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138"
CMD="${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS} ${ARCH}"

$CMD &>${FRONTNODE_LOGS}/${OUTFILE}
