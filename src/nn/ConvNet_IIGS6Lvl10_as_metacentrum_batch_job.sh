#!/bin/bash
#PBS -N ConvNet_IIGS6Lvl10_single_sample_experiments
#PBS -q gpu
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script generates dataset for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  qsub ConvNet_IIGS6Lvl10_as_metacentrum_batch_job.sh

# configure variables
FRONTNODE_HOME="/storage/plzen1/home/mathemage"
REPO_DIR="${FRONTNODE_HOME}/beyond-deepstack/TensorCFR"
FRONTNODE_LOGS="${FRONTNODE_HOME}/beyond-deepstack/logs/ConvNet_IIGS6Lvl10_single_sample_experiments"
WORKER_LOGS="${SCRATCHDIR}/TensorCFR/logs"

trap 'clean_scratch' TERM EXIT  # nastaveni uklidu SCRATCHE v pripade chyby
module add tensorflow-1.7.1-gpu-python3

# move to Metacentrum's temporal drive
cd ${SCRATCHDIR} || exit 1
cp -r ${REPO_DIR} .           # copy repo from FRONTNODE

# run TensorCFR
cd TensorCFR
sh ./nn_iigs6_single_sample_experiments.sh &>logs/nn_iigs6_single_sample_experiments.out

# copy results back from temporal drive
mkdir -p ${FRONTNODE_LOGS}
mv ${WORKER_LOGS}/* ${FRONTNODE_LOGS} || export CLEAN_SCRATCH=false # presune vysledky do domovskeho adresare nebo je ponecha ve scratchi v pripade chyby

