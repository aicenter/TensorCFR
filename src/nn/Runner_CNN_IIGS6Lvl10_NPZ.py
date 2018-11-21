#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import logging

from src.nn.AbstractNNRunner import AbstractNNRunner
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ
from src.utils.other_utils import activate_script


class Runner_CNN_IIGS6Lvl10_NPZ(AbstractNNRunner):
	@property
	def default_extractor_arch(self):
		return "C-{}".format(ConvNet_IIGS6Lvl10.INPUT_FEATURES_DIM)

	@property
	def default_regressor_arch(self):
		return "C-{}".format(ConvNet_IIGS6Lvl10.INPUT_FEATURES_DIM)

	def add_arguments_to_argparser(self):
		self.argparser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
		self.argparser.add_argument("--dataset_directory", default="data/IIGS6Lvl10/minimal_dataset/2",
		                            help="Relative path to dataset folder.")
		self.argparser.add_argument("--extractor", default=self.default_extractor_arch, type=str,
		                            help="Description of the feature extactor architecture.")
		self.argparser.add_argument("--regressor", default=self.default_regressor_arch, type=str,
		                            help="Description of the value regressor architecture.")
		self.argparser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
		self.argparser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
		self.argparser.add_argument("--ckpt_every", default=2, type=float, help="Checkpoint every `ckpt_every` epochs.")
		self.argparser.add_argument("--ckpt_dir", default=None, type=str, help="Checkpoint directory with model to restore.")
		self.argparser.add_argument("--ckpt_basename", default=None, type=str, help="Checkpoint name with model to restore.")

	@staticmethod
	def datasets_from_npz(dataset_directory, script_directory):
		npz_basename = "IIGS6_1_6_false_true_lvl10"
		trainset = DatasetFromNPZ("{}/{}/{}_train.npz".format(script_directory, dataset_directory, npz_basename))
		devset = DatasetFromNPZ("{}/{}/{}_dev.npz".format(script_directory, dataset_directory, npz_basename))
		testset = DatasetFromNPZ("{}/{}/{}_test.npz".format(script_directory, dataset_directory, npz_basename))
		return devset, testset, trainset

	def init_datasets(self, dataset_directory):
		import os
		script_directory = os.path.dirname(os.path.abspath(__file__))
		devset, testset, trainset = Runner_CNN_IIGS6Lvl10_NPZ.datasets_from_npz(dataset_directory, script_directory)
		return devset, testset, trainset

	def construct_network(self):
		self.network = ConvNet_IIGS6Lvl10(threads=self.args.threads, fixed_randomness=self.fixed_randomness)
		self.network.construct(self.args)
		return self.network


if __name__ == '__main__' and activate_script():
	logging.basicConfig(
		format='%(asctime)s %(message)s',
		level=logging.INFO
	)
	runner = Runner_CNN_IIGS6Lvl10_NPZ()
	runner.run_neural_net(ckpt_every=2)

	runner_from_ckpt = Runner_CNN_IIGS6Lvl10_NPZ()
	runner_from_ckpt.run_neural_net_from_ckpt(ckpt_dir=runner.args.logdir, ckpt_basename=runner.ckpt_basenames[-1])
