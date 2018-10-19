#!/bin/bash
#PBS -N II-GS6_50_data_pt
#PBS -q gpu
#PBS -l walltime=5:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script generates dataset for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  qsub generate_50_datapoints_of_IIGS6_6cards_as_metacentrum_batch_job.sh

# configure variables
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
FRONTNODE_DATA="${FRONTNODE_HOME}/beyond-deepstack/data/IIGS6/50_datapoints"
DATASET_DIR="${SCRATCHDIR}/TensorCFR/src/nn/data/out/IIGS6/50_datapoints"

trap 'clean_scratch' TERM EXIT  # nastaveni uklidu SCRATCHE v pripade chyby
module add tensorflow-1.7.1-gpu-python3

# move to Metacentrum's temporal drive
cd ${SCRATCHDIR} || exit 1
cp -r ${REPO_DIR} .           # copy repo from FRONTNODE

# run TensorCFR
cd TensorCFR
python -m src.nn.data.generate_data_of_IIGS6

# copy results back from temporal drive
mkdir -p ${FRONTNODE_DATA}
mv ${DATASET_DIR}/* ${FRONTNODE_DATA} || export CLEAN_SCRATCH=false # presune vysledky do domovskeho adresare nebo je ponecha ve scratchi v pripade chyby

