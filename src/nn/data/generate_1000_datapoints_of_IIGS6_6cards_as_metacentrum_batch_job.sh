#!/bin/bash
#PBS -N II-GS6_50_data_pt
#PBS -q gpu
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script generates dataset for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  qsub generate_1000_datapoints_of_IIGS6_6cards_as_metacentrum_batch_job.sh

# configure variables
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
FRONTNODE_DATA="${FRONTNODE_HOME}/beyond-deepstack/data/IIGS6/1000_datapoints"
DATASET_DIR="TensorCFR/src/nn/data/out/IIGS6/1000_datapoints" # TODO rename the current script with 1000 datapoints

trap 'clean_scratch' TERM EXIT  # nastaveni uklidu SCRATCHE v pripade chyby
module add tensorflow-1.7.1-gpu-python3

cd ${SCRATCHDIR} || exit 1
cp ${REPO_DIR} .           # copy repo from FRONTNODE

# run TensorCFR
cd TensorCFR
python -m src.algorithms.tensorcfr_flattened_domains.generate_data_of_IIGS6

# copy results back from temporal drive
mkdir ${FRONTNODE_DATA}
mv ${DATASET_DIR}/* ${FRONTNODE_DATA} || export CLEAN_SCRATCH=false # presune vysledky do domovskeho adresare nebo je ponecha ve scratchi v pripade chyby

# clean up
cd ${SCRATCHDIR}
rm -Rf *
