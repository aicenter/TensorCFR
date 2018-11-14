#!/usr/bin/env bash
PREFIX_PATH='/home/ruda/Documents/Projects/tensorcfr/TensorCFR/logs_KAREL'

python -m src.utils.reading_tf_log train/loss pic.png  --title 'Train/Loss' --xlabel 'Steps' --ylabel 'Error' --graphs_names 'NN1' 'NN2' 'NN3' 'NN4' 'NN5' \
--tensorflow_log \
$PREFIX_PATH/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021744-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,r=C-138,C-138,C-138,t=1,tr=0.8/events.out.tfevents.1541899082.glados10.cerit-sc.cz.v2 \
$PREFIX_PATH/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021752-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,r=C-138,C-138,t=1,tr=0.8/events.out.tfevents.1541899086.white1.cerit-sc.cz.v2 \
$PREFIX_PATH/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_021836-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,r=C-138,t=1,tr=0.8/events.out.tfevents.1541899127.white1.cerit-sc.cz.v2 \
$PREFIX_PATH/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_030427-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8/events.out.tfevents.1541901887.glados16.cerit-sc.cz.v2 \
$PREFIX_PATH/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_133801-bs=32,ce=64.0,dr=0.1,e=512,e=C-46,C-46,C-46,C-46,C-46,C-46,C-46,r=C-138,C-138,C-138,C-138,C-138,C-138,C-138,t=1,tr=0.8/events.out.tfevents.1541939902.konos1.fav.zcu.cz.v2
