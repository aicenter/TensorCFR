#!/bin/bash
#PBS -N ConvNet_IIGS6Lvl10_1xlayers_single_sample_experiments
#PBS -q gpu
#PBS -l walltime=4:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script runs ConvNet_IIGS6Lvl10 with single_sample datasets for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  qsub ConvNet_IIGS6Lvl10_as_metacentrum_batch_job.sh

# configure variables
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
EXPERIMENT_NAME="ConvNet_IIGS6Lvl10_single_sample_experiments"
FRONTNODE_LOGS="${REPO_DIR}/logs/${EXPERIMENT_NAME}"
OUTFILE=${EXPERIMENT_NAME}_$(date -d "today" +"%Y%m%d%H%M").out

mkdir -p ${FRONTNODE_LOGS}

module add tensorflow-1.7.1-gpu-python3

# run TensorCFR
cd ${REPO_DIR} || exit 1
PYTHON=python
COMMON_ARGS="--dataset_directory data/IIGS6Lvl10/minimal_dataset/2 --epochs 30000"
ARCH="--extractor C-46 --regressor C-138"
CMD="${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS} ${ARCH}"

$CMD &>${FRONTNODE_LOGS}/${OUTFILE}
