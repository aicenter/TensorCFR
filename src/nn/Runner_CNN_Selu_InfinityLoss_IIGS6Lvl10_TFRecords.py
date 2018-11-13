#!/usr/bin/env python3

from src.nn.ConvNet_Selu_InfinityLoss_IIGS6Lvl10 import ConvNet_Selu_InfinityLoss_IIGS6Lvl10
# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
from src.nn.Runner_CNN_IIGS6Lvl10_TFRecords import Runner_CNN_IIGS6Lvl10_TFRecords


class Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_TFRecords(Runner_CNN_IIGS6Lvl10_TFRecords):
	def construct_network(self):
		network = ConvNet_Selu_InfinityLoss_IIGS6Lvl10(threads=self.args.threads, fixed_randomness=self.fixed_randomness)
		print("network = ConvNet_Selu_InfinityLoss_IIGS6Lvl10(threads=self.args.threads, "
		      "fixed_randomness=self.fixed_randomness)")
		network.construct(self.args)
		print("network.construct(self.args)")
		return network


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True

if __name__ == '__main__' and ACTIVATE_FILE:
	runner = Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_TFRecords()
	print("runner = Runner_CNN_Selu_InfinityLoss_IIGS6Lvl10_TFRecords()")
	runner.run_neural_net()
	print("runner.run_neural_net()")
