#!/bin/bash
#PBS -N II-GS6_50_data_pt
#PBS -q gpu
#PBS -l walltime=2:00:00
#PBS -l select=1:ncpus=10:ngpus=1:gpu_cap=cuda35:mem=30gb:scratch_local=10gb

# README
# This script generates dataset for II-GS6 on Metacentrum's server.
#
# Run this command in Metacentrum's command line to run the job:
#  qsub generate_data_of_IIGS6_6cards_metacentrum_batch_job.sh

# configure variables
HOMEDIR=/storage/plzen1/home/${PBS_O_LOGNAME}
DATADIR=${HOMEDIR}/tensorcfr
DATADIR_OUTPUT=${DATADIR}/output/job_${PBS_JOBID}

trap 'clean_scratch' TERM EXIT  # nastaveni uklidu SCRATCHE v pripade chyby
cp ${HOMEDIR}/.ssh/* ./.ssh/    # copy SSH keys for `git clone` without password
module add tensorflow-1.7.1-gpu-python3

cd ${SCRATCHDIR} || exit 1 # vstoupi do scratch adresare
git clone git@gitlab.com:beyond-deepstack/TensorCFR.git

# run TensorCFR
cd TensorCFR
python -m src.algorithms.tensorcfr_flattened_domains.tensorcfr_on_goofspiel6

mkdir ${DATADIR_OUTPUT}
mv out/*.csv ${DATADIR_OUTPUT} || export CLEAN_SCRATCH=false # presune vysledky do domovskeho adresare nebo je ponecha ve scratchi v pripade chyby

# clean up
cd ${SCRATCHDIR}
rm -Rf *
rm -Rf ./.ssh/  # remove SSH credentials