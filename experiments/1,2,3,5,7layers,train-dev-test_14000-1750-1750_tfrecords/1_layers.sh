#!/usr/bin/env bash
#PBS -N CNN_IIGS6Lvl10_1xlayers_17500seeds_tfrecords
#PBS -q gpu
#PBS -l walltime=02:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script runs CNN_IIGS6Lvl10 with datasets of 17500 seed files stored in TFRecords for II-GS6 on Metacentrum's
# server.
#
# Run this command in Metacentrum's command line to run the job:
#  submit_all.sh

# configure variables
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
EXPERIMENT_NAME="CNN_IIGS6Lvl10_17500seeds_tfrecords_experiments"
FRONTNODE_LOGS="${REPO_DIR}/logs/${EXPERIMENT_NAME}"
OUTFILE=${EXPERIMENT_NAME}_$(date -d "today" +"%Y%m%d%H%M").out

mkdir -p ${FRONTNODE_LOGS}

module add tensorflow-1.7.1-gpu-python3

cd ${REPO_DIR} || exit 1
PYTHON=python3
#DATASET_DIRECTORY="../data/IIGS6/17450_datapoints_1_seed_per_file/tfrecord_dataset_IIGS6_1_6_false_true_lvl10"
DATASET_DIRECTORY="../data/IIGS6/17450_datapoints_128_seed_per_file/tfrecord_dataset_IIGS6_1_6_false_true_lvl10"
COMMON_ARGS="--dataset_directory ${DATASET_DIRECTORY} --epochs 16 --ckpt_every 4"
ARCH="--extractor C-46 --regressor C-138"
CMD="${PYTHON} -m src.nn.Runner_CNN_IIGS6Lvl10_TFRecords ${COMMON_ARGS} ${ARCH}"

echo ${CMD} |& tee ${FRONTNODE_LOGS}/${OUTFILE}
${CMD} |& tee ${FRONTNODE_LOGS}/${OUTFILE}