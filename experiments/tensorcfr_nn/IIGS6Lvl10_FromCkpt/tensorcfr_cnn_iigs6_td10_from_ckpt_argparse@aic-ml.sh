#!/usr/bin/env bash

PYTHON=python3.6
BASENAME=tensorcfr_CNN_IIGS6_td10_from_ckpt
SCRIPT=src.algorithms.tensorcfr_nn.${BASENAME}
COMMON_ARGS=""
CKPT_COMMON_DIR="/home/mathemage/beyond-deepstack/TensorCFR/src/algorithms/tensorcfr_nn/checkpoints"        # TODO?
OUTDIR=/home/mathemage/beyond-deepstack/TensorCFR/experiments/tensorcfr_nn/IIGS6Lvl10_FromCkpt
OUTFILE=${OUTDIR}/${BASENAME}.out

# 1 layers
CKPT_DIR="Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021836-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,r=C-138,t=1,tr=0.8"
CKPT_BASENAME="final_2018-11-11_07:46:15.ckpt"
CKPTS="--ckpt_dir ${CKPT_COMMON_DIR} --ckpt_basename ${CKPT_BASENAME}"
ARCH="--extractor C-46 --regressor C-138"
RUNS=10
for i in `seq 10` ; do
    echo CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} ${CKPTS} |& tee ${OUTFILE}.1_layers.run${i}
done

## 5 layers
#CKPT_DIR="Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_030427-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8"
#CKPT_BASENAME="final_2018-11-11_16:52:00.ckpt"
#CKPTS="--ckpt_dir ${CKPT_COMMON_DIR} --ckpt_basename ${CKPT_BASENAME}"
#ARCH="--extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138"
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} ${CKPTS} |& tee ${OUTFILE}.5_layers
#
### 7-layers
#CKPT_DIR="src/algorithms/tensorcfr_nn/checkpoints/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_133801-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8"
#CKPT_BASENAME="final_2018-11-12_09:37:52.ckpt"
#CKPTS="--ckpt_dir ${CKPT_COMMON_DIR} --ckpt_basename ${CKPT_BASENAME}"
#ARCH="--extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138"
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m ${SCRIPT} ${COMMON_ARGS} ${ARCH} ${CKPTS} |& tee ${OUTFILE}.7_layers
