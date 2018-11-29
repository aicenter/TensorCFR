#!/usr/bin/env bash
PREFIX_PATH='./src/utils/plots/example_data'

python -m src.utils.plots.tensorflow_log train/loss pic.png  --title 'Train/Loss' --xlabel 'Steps' --ylabel 'Error' --curves_names 'NN1' 'NN2' \
--tensorflow_log \
$PREFIX_PATH/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021744-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,r=C-138,C-138,C-138,t=1,tr=0.8/events.out.tfevents \
$PREFIX_PATH/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021752-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,r=C-138,C-138,t=1,tr=0.8/events.out.tfevents
