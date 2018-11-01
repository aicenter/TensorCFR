#!/bin/bash
<<<<<<< HEAD
#PBS -N ConvNet_IIGS6Lvl10_1xlayers_batch512_units500_900seeds_npz
=======
#PBS -N ConvNet_IIGS6Lvl10_1xlayers_900seeds_npz
>>>>>>> 209ae84... Copy `7layers,width500,batchsize512,train-dev-test_720-90-90`
#PBS -q gpu
#PBS -l walltime=24:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script runs ConvNet_IIGS6Lvl10 with datasets of 900 seed files stored in NPZ for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  metacentrum.sh

# configure variables
<<<<<<< HEAD
FRONTNODE_HOME="/storage/plzen1/home/seitzdom"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
EXPERIMENT_NAME="ConvNet_IIGS6Lvl10__batch512_units500_900seeds_npz_experiments"
=======
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
EXPERIMENT_NAME="ConvNet_IIGS6Lvl10_900seeds_npz_experiments"
>>>>>>> 209ae84... Copy `7layers,width500,batchsize512,train-dev-test_720-90-90`
FRONTNODE_LOGS="${REPO_DIR}/logs/${EXPERIMENT_NAME}"
OUTFILE=${EXPERIMENT_NAME}_$(date -d "today" +"%Y%m%d%H%M").out

mkdir -p ${FRONTNODE_LOGS}

module add tensorflow-1.7.1-gpu-python3

# run TensorCFR
cd ${REPO_DIR} || exit 1
PYTHON=python
DATASET_DIRECTORY="../../../data/IIGS6/IIGS6_1_6_false_true_lvl10_npz_900_seeds"
<<<<<<< HEAD
COMMON_ARGS="--dataset_directory ${DATASET_DIRECTORY} --epochs 25"
ARCH="--extractor C-500 --regressor C-1500"
=======
COMMON_ARGS="--dataset_directory ${DATASET_DIRECTORY} --epochs 25000"
ARCH="--extractor C-46 --regressor C-138"
>>>>>>> 209ae84... Copy `7layers,width500,batchsize512,train-dev-test_720-90-90`
CMD="${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_NPZ ${COMMON_ARGS} ${ARCH}"

$CMD &>${FRONTNODE_LOGS}/${OUTFILE}
