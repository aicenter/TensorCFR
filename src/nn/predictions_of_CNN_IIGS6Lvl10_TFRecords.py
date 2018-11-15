#!/usr/bin/env python3
import logging

from src.nn.Runner_CNN_IIGS6Lvl10_TFRecords import Runner_CNN_IIGS6Lvl10_TFRecords

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
	logging.basicConfig(
		format='%(asctime)s %(message)s',
		level=logging.INFO
		# level=logging.DEBUG
	)

	runner_from_ckpt = Runner_CNN_IIGS6Lvl10_TFRecords()
	# Note: you can test this on:
	# --ckpt_dir "logs/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_204734-bs=32,ce=2,dr=0.1,e=3,e=C-46,r=C-46,t=1,tr=0.8"
	# --ckpt_basename "final_2018-11-11_20:47:52.ckpt"
	runner_from_ckpt.run_neural_net_from_ckpt()
