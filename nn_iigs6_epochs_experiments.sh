#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1 python3.6 -m src.nn.ConvNet_IIGS6Lvl10 --epochs 500
# CUDA_VISIBLE_DEVICES=1 python3.6 -m src.nn.ConvNet_IIGS6Lvl10 --epochs 5000
# CUDA_VISIBLE_DEVICES=1 python3.6 -m src.nn.ConvNet_IIGS6Lvl10 --epochs 50000
# CUDA_VISIBLE_DEVICES=1 python3.6 -m src.nn.ConvNet_IIGS6Lvl10 --epochs 500000

# medium-size dataset: 210 seed files
#CUDA_VISIBLE_DEVICES=1 python3.6 -m src.nn.ConvNet_IIGS6Lvl10 --epochs 500
#CUDA_VISIBLE_DEVICES=1 python3.6 -m src.nn.ConvNet_IIGS6Lvl10 --epochs 5000
#CUDA_VISIBLE_DEVICES=1 python3.6 -m src.nn.ConvNet_IIGS6Lvl10 --epochs 50000

# baselines (on local)
PYTHON=/usr/bin/python3
${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --epochs 50
${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --epochs 500
${PYTHON} -m src.nn.ConvNetBaseline_IIGS6Lvl10 --epochs 5000
