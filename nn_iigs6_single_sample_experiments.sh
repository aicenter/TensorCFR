#!/usr/bin/env bash

# @aic-ml: 1-file dataset
PYTHON=python3.6
COMMON_ARGS="--dataset_directory data/IIGS6Lvl10/minimal_dataset/2 --epochs 15000"
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS}--extractor C-46 --regressor C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS}--extractor C-46,C-46 --regressor C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS}--extractor C-46,C-46,C-46 --regressor C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS}--extractor C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS}--extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS}--extractor C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNet_IIGS6Lvl10 ${COMMON_ARGS}--extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138
