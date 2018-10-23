#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
from abc import abstractmethod

import numpy as np

from src.commons.constants import SEED_FOR_TESTING


class AbstractNNRunner:
	def __init__(self, fixed_randomness=False):
		self.fixed_randomness = fixed_randomness

	@property
	def default_extractor_arch(self):
		raise NotImplementedError

	@property
	def default_regressor_arch(self):
		raise NotImplementedError

	def parse_arguments(self):
		import argparse

		parser = argparse.ArgumentParser()
		parser.add_argument("--batch_size", default=32, type=int, help="Batch size.")
		parser.add_argument("--dataset_directory", default="data/IIGS6Lvl10/minimal_dataset/2",
		                    help="Relative path to dataset folder.")
		parser.add_argument("--extractor", default=self.default_extractor_arch, type=str,
		                    help="Description of the feature extactor architecture.")
		parser.add_argument("--regressor", default=self.default_regressor_arch, type=str,
		                    help="Description of the value regressor architecture.")
		parser.add_argument("--epochs", default=5, type=int, help="Number of epochs.")
		parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

		args = parser.parse_args()
		print("args: {}".format(args))
		return args

	@staticmethod
	def create_logdir(args):
		import datetime
		import re
		import os

		del args.dataset_directory
		args.logdir = "logs/{}-{}-{}".format(
			os.path.basename(__file__),
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
		)
		if not os.path.exists("logs"):
			os.mkdir("logs")  # TF 1.6 will do this by itself
		return args

	@abstractmethod
	def init_datasets(self, dataset_directory):
		pass

	@abstractmethod
	def construct_network(self, args):
		pass

	@staticmethod
	def train_one_epoch(args, network, trainset):
		while not trainset.epoch_finished():
			reaches, targets = trainset.next_batch(args.batch_size)
			network.train(reaches, targets)

	@staticmethod
	def evaluate_devset(devset, epoch, network):
		devset_error_mse, devset_error_infinity = network.evaluate("dev", devset.features, devset.targets)
		print("[epoch #{}] dev MSE {}, \tdev L-infinity error {}".format(epoch, devset_error_mse, devset_error_infinity))

	@staticmethod
	def evaluate_testset(network, testset):
		testset_error_mse, testset_error_infinity = network.evaluate("test", testset.features, testset.targets)
		print()
		print("mean squared error on testset: {}".format(testset_error_mse))
		print("L-infinity error on testset: {}".format(testset_error_infinity))

	@staticmethod
	def showcase_predictions(network, trainset):
		print()
		print("Predictions of initial 2 training examples:")
		print(network.predict(trainset.features[:2]))

	def run_neural_net(self):
		np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
		if self.fixed_randomness:
			print("Abstract: self.fixed_randomness is {}".format(self.fixed_randomness))
			np.random.seed(SEED_FOR_TESTING)  # Fix random seed

		args = self.parse_arguments()
		dataset_directory = args.dataset_directory
		args = AbstractNNRunner.create_logdir(args)

		devset, testset, trainset = self.init_datasets(dataset_directory)
		network = self.construct_network(args)

		for epoch in range(args.epochs):
			AbstractNNRunner.train_one_epoch(args, network, trainset)
			AbstractNNRunner.evaluate_devset(devset, epoch, network)

		AbstractNNRunner.evaluate_testset(network, testset)
		AbstractNNRunner.showcase_predictions(network, trainset)
