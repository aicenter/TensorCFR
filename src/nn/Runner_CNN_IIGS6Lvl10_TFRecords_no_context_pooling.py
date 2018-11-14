#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
from src.nn.Runner_CNN_IIGS6Lvl10_TFRecords import Runner_CNN_IIGS6Lvl10_TFRecords

# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


class Runner_CNN_IIGS6Lvl10_TFRecords_no_context_pooling(Runner_CNN_IIGS6Lvl10_TFRecords):
	pass


if __name__ == '__main__' and ACTIVATE_FILE:
	runner = Runner_CNN_IIGS6Lvl10_TFRecords_no_context_pooling()
	runner.run_neural_net(ckpt_every=2)

	runner_from_ckpt = Runner_CNN_IIGS6Lvl10_TFRecords_no_context_pooling()
	# runner_from_ckpt.run_neural_net_from_ckpt(ckpt_dir=runner.args.logdir, ckpt_basename=runner.ckpt_basenames[-1])

	# Note: you can test this on:
	# i.e.
	# --ckpt_dir "logs/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_204734-bs=32,ce=2,dr=0.1,e=3,e=C-46,r=C-46,t=1,tr=0.8"
	# --ckpt_basename "final_2018-11-11_20:47:52.ckpt successful"
	runner_from_ckpt.run_neural_net_from_ckpt()
