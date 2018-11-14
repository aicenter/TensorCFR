#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import numpy as np
import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ

FIXED_RANDOMNESS = False


class ConvNet_Selu_MSELoss_IIGS6Lvl10(ConvNet_IIGS6Lvl10):

	def construct_loss(self):
		with tf.variable_scope("loss"):
			self.loss = self.mean_squared_error
			print("loss MSE constructed")
			self.print_operations_count()

		def construct_feature_extractor(self, args):
			"""
			Add layers described in the args.extractor. Layers are separated by a comma.

			:param args: Arguments from commandline
			:return:
			"""
			with tf.variable_scope("extractor"):
				extractor_desc = args.extractor.split(',')
				for l, layer_desc in enumerate(extractor_desc):
					specs = layer_desc.split('-')
					layer_name = "extractor{}_{}".format(l, layer_desc)
					# - C-hidden_layer_size: 1D convolutional with SeLU activation and specified output size (channels). Ex: "C-100"
					if specs[0] == 'C':
						self.latest_layer = tf.layers.conv1d(
							inputs=self.latest_layer,
							filters=int(specs[1]),
							kernel_size=1,
							kernel_initializer=tf.keras.initializers.lecun_normal(),
							activation=tf.nn.selu,
							data_format="channels_first",
							name=layer_name
						)
						print("{} constructed".format(layer_name))
					else:
						raise ValueError("Invalid extractor specification '{}'".format(specs))

		def construct_value_regressor(self, args):
			"""
			Add layers described in the args.regressor. Layers are separated by a comma.

			:param args: Arguments from commandline
			:return:
			"""
			with tf.variable_scope("regressor"):
				regressor_desc = args.regressor.split(',')
				for l, layer_desc in enumerate(regressor_desc):
					specs = layer_desc.split('-')
					layer_name = "regressor{}_{}".format(l, layer_desc)
					# - C-hidden_layer_size: 1D convolutional with ReLU activation and specified output size (channels). Ex: "C-100"
					if specs[0] == 'C':
						self.latest_layer = tf.layers.conv1d(
							inputs=self.latest_layer,
							filters=int(specs[1]),
							kernel_size=1,
							kernel_initializer=tf.keras.initializers.lecun_normal(),
							activation=tf.nn.selu,
							data_format="channels_first",
							name=layer_name
						)
						print("{} constructed".format(layer_name))
					else:
						raise ValueError("Invalid regressor specification '{}'".format(specs))


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
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
	parser.add_argument("--extractor", default="C-{}".format(ConvNet_Selu_InfinityLoss_IIGS6Lvl10.INPUT_FEATURES_DIM), type=str,
	                    help="Description of the feature extactor architecture.")
	parser.add_argument("--regressor", default="C-{}".format(ConvNet_Selu_InfinityLoss_IIGS6Lvl10.INPUT_FEATURES_DIM), type=str,
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
	network = ConvNet_Selu_MSELoss_IIGS6Lvl10(threads=args.threads)
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
