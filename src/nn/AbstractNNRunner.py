#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import argparse
from abc import abstractmethod

import numpy as np

from src.commons.constants import SEED_FOR_TESTING


class AbstractNNRunner:
	def __init__(self, fixed_randomness=False):
		self.argparser = argparse.ArgumentParser()
		self.fixed_randomness = fixed_randomness
		self.args = None
		self.network = None
		self.epoch = None

	@property
	def default_extractor_arch(self):
		raise NotImplementedError

	@property
	def default_regressor_arch(self):
		raise NotImplementedError

	def add_arguments_to_argparser(self):
		raise NotImplementedError

	def parse_arguments(self):
		self.add_arguments_to_argparser()
		self.args = self.argparser.parse_args()
		print("args: {}".format(self.args))

	def create_logdir(self):
		import datetime
		import re
		import os

		del self.args.dataset_directory
		self.args.logdir = "logs/{}-{}-{}".format(
			os.path.basename(self.__class__.__name__),
			datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
			",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value)
			          for key, value in sorted(vars(self.args).items())))
		)
		if not os.path.exists("logs"):
			os.mkdir("logs")  # TF 1.6 will do this by itself

	@abstractmethod
	def init_datasets(self, dataset_directory):
		pass

	@abstractmethod
	def construct_network(self):
		pass

	def train_one_epoch(self, trainset):
		while not trainset.epoch_finished():
			print("[epoch #{}, batch #{}] Training...".format(self.epoch, trainset.batch_id))
			reaches, targets = trainset.next_batch(self.args.batch_size)
			self.network.train(reaches, targets)

	def evaluate_devset(self, devset):
		devset_error_mse, devset_error_infinity = self.network.evaluate("dev", devset.features, devset.targets)
		print("[epoch #{}] dev MSE {}, \tdev L-infinity error {}".format(self.epoch, devset_error_mse, devset_error_infinity))

	def evaluate_testset(self, testset):
		testset_error_mse, testset_error_infinity = self.network.evaluate("test", testset.features, testset.targets)
		print()
		print("mean squared error on testset: {}".format(testset_error_mse))
		print("L-infinity error on testset: {}".format(testset_error_infinity))

	def showcase_predictions(self, trainset):
		print()
		print("Predictions of initial 2 training examples:")
		print(self.network.predict(trainset.features[:2]))

	def run_neural_net(self):
		np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
		if self.fixed_randomness:
			print("Abstract: self.fixed_randomness is {}".format(self.fixed_randomness))
			np.random.seed(SEED_FOR_TESTING)  # Fix random seed

		self.parse_arguments()
		dataset_directory = self.args.dataset_directory
		self.create_logdir()

		devset, testset, trainset = self.init_datasets(dataset_directory)
		self.network = self.construct_network()

		for self.epoch in range(self.args.epochs):
			self.train_one_epoch(trainset)
			self.evaluate_devset(devset)

		self.evaluate_testset(testset)
		self.showcase_predictions(trainset)
