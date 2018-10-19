#!/usr/bin/env bash

# @aic-ml: 1-file dataset, baselines
PYTHON=python3.6
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --extractor C-46 --regressor C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --extractor C-46,C-46 --regressor C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --extractor C-46,C-46,C-46 --regressor C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --extractor C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --extractor C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --extractor C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138
CUDA_VISIBLE_DEVICES=1 ${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --extractor C-46,C-46,C-46,C-46,C-46,C-46,C-46 --regressor C-138,C-138,C-138,C-138,C-138,C-138,C-138
