#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py

from src.nn.Runner_CNN_IIGS6Lvl10_NPZ import Runner_CNN_IIGS6Lvl10_NPZ
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ


class Runner_CNN_IIGS6Lvl10_TFRecords(Runner_CNN_IIGS6Lvl10_NPZ):   # TODO test parent here
	@staticmethod
	def datasets_from_tfrecords(dataset_directory, script_directory):
		npz_basename = "IIGS6_1_6_false_true_lvl10"
		trainset = DatasetFromNPZ("{}/{}/{}_train.npz".format(script_directory, dataset_directory, npz_basename))
		devset = DatasetFromNPZ("{}/{}/{}_dev.npz".format(script_directory, dataset_directory, npz_basename))
		testset = DatasetFromNPZ("{}/{}/{}_test.npz".format(script_directory, dataset_directory, npz_basename))
		return devset, testset, trainset

	def init_datasets(self, dataset_directory):
		import os
		script_directory = os.path.dirname(os.path.abspath(__file__))
		devset, testset, trainset = Runner_CNN_IIGS6Lvl10_TFRecords.datasets_from_tfrecords(dataset_directory, script_directory)
		return devset, testset, trainset


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = True


if __name__ == '__main__' and ACTIVATE_FILE:
	runner = Runner_CNN_IIGS6Lvl10_TFRecords()
	runner.run_neural_net()
