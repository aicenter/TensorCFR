#!/usr/bin/env bash

# @aic-ml
PYTHON=python3.6
#CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.features.goofspiel.IIGS6.tf_datasets_IIGS6_1_6_false_true_lvl10
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.features.goofspiel.IIGS6.tfrecords_reaches_values_IIGS6_1_6_false_true_lvl10
