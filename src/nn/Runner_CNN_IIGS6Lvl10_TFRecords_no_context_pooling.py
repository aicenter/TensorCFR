#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
from src.nn.FullyConvNet_IIGS6Lvl10_no_context_pooling import FullyConvNet_IIGS6Lvl10_no_context_pooling
from src.nn.Runner_CNN_IIGS6Lvl10_TFRecords import Runner_CNN_IIGS6Lvl10_TFRecords
from src.utils.other_utils import activate_script


class Runner_CNN_IIGS6Lvl10_TFRecords_no_context_pooling(Runner_CNN_IIGS6Lvl10_TFRecords):
	def construct_network(self):
		self.network = FullyConvNet_IIGS6Lvl10_no_context_pooling(threads=self.args.threads, fixed_randomness=self.fixed_randomness)
		self.network.construct(self.args)
		return self.network


if __name__ == '__main__' and activate_script():
	runner = Runner_CNN_IIGS6Lvl10_TFRecords_no_context_pooling()
	runner.run_neural_net(ckpt_every=2)

	# TODO modify
	# runner_from_ckpt = Runner_CNN_IIGS6Lvl10_TFRecords_no_context_pooling()
	# runner_from_ckpt.run_neural_net_from_ckpt(ckpt_dir=runner.args.logdir, ckpt_basename=runner.ckpt_basenames[-1])

	# TODO modify
	# Note: you can test this on:
	# i.e.
	# --ckpt_dir "logs/Runner_CNN_IIGS6Lvl10_TFRecords-2018-11-11_204734-bs=32,ce=2,dr=0.1,e=3,e=C-46,r=C-46,t=1,tr=0.8"
	# --ckpt_basename "final_2018-11-11_20:47:52.ckpt successful"
	# runner_from_ckpt.run_neural_net_from_ckpt()
