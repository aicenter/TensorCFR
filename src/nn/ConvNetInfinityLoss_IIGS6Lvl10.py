#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import numpy as np
import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ
from src.utils.other_utils import activate_script

FIXED_RANDOMNESS = False


class ConvNetInfinityLoss_IIGS6Lvl10(ConvNet_IIGS6Lvl10):
	def construct_loss(self):
		with tf.variable_scope("loss"):
			self.loss = self.huber_loss + self.l_infinity_error
			print("loss (Huber + L-infinity) constructed")
			self.print_operations_count()


if __name__ == '__main__' and activate_script():
	import argparse
	import datetime
	import os
	import re

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	if FIXED_RANDOMNESS:
		np.random.seed(SEED_FOR_TESTING)  # Fix random seed

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
	parser.add_argument("--extractor", default="C-{}".format(ConvNetInfinityLoss_IIGS6Lvl10.INPUT_FEATURES_DIM), type=str,
	                    help="Description of the feature extactor architecture.")
	parser.add_argument("--regressor", default="C-{}".format(ConvNetInfinityLoss_IIGS6Lvl10.INPUT_FEATURES_DIM), type=str,
	                    help="Description of the value regressor architecture.")
	parser.add_argument("--epochs", default=50, type=int, help="Number of epochs.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

	args = parser.parse_args()
	print("args: {}".format(args))

	# Create logdir name
	args.logdir = "logs/{}-{}-{}".format(
		os.path.basename(__file__),
		datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
		",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
	)
	if not os.path.exists("logs"):
		os.mkdir("logs")  # TF 1.6 will do this by itself

	# Load the data
	script_directory = os.path.dirname(os.path.abspath(__file__))
	dataset_directory = "data/IIGS6Lvl10/80-10-10"
	npz_basename = "IIGS6_1_6_false_true_lvl10"
	trainset = DatasetFromNPZ("{}/{}/{}_train.npz".format(script_directory, dataset_directory, npz_basename))
	devset = DatasetFromNPZ("{}/{}/{}_dev.npz".format(script_directory, dataset_directory, npz_basename))
	testset = DatasetFromNPZ("{}/{}/{}_test.npz".format(script_directory, dataset_directory, npz_basename))

	# Construct the network
	network = ConvNetInfinityLoss_IIGS6Lvl10(threads=args.threads)
	network.construct(args)

	# Train
	for epoch in range(args.epochs):
		while not trainset.epoch_finished():
			features, targets = trainset.next_batch(args.batch_size)
			network.train(features, targets)

		# Evaluate on development set
		devset_error_mse, devset_error_infinity = network.evaluate("dev", devset.features, devset.targets)
		print("[epoch #{}] dev MSE {}, \tdev L-infinity error {}".format(epoch, devset_error_mse, devset_error_infinity))

	# Evaluate on test set
	testset_error_mse, testset_error_infinity = network.evaluate("test", testset.features, testset.targets)
	print()
	print("mean squared error on testset: {}".format(testset_error_mse))
	print("L-infinity error on testset: {}".format(testset_error_infinity))

	print()
	print("Predictions of initial 2 training examples:")
	print(network.predict(trainset.features[:2]))
