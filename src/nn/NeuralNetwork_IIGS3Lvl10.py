#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import numpy as np
import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING, FLOAT_DTYPE
from src.nn.data.DatasetFromNPZ import DatasetFromNPZ
from src.nn.features.goofspiel.IIGS3.node_to_public_states_IIGS3_1_3_false_true_lvl7 import get_node_to_public_state

FIXED_RANDOMNESS = False


class NeuralNetwork_IIGS3Lvl10:
	NUM_NODES = 36
	FEATURES_DIM = 3 * (2 + 2 + 2) + 1  # 6x 1-of-3-hot encodings (3 per hierarchy) + reach probability
	TARGETS_DIM = 1
	NUM_PUBLIC_STATES = 3 ** 2              # i.e. 3^rounds

	def __init__(self, threads, seed=SEED_FOR_TESTING):
		# Create an empty graph and a session
		graph = tf.Graph()
		if FIXED_RANDOMNESS:
			graph.seed = seed
			self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
			                                                             intra_op_parallelism_threads=threads))
		else:
			self.session = tf.Session(graph=graph)
		self._node_to_public_state = get_node_to_public_state()
		print("node_to_public_state:\n{}".format(self._node_to_public_state))

	def construct_feature_extractor(self, args):
		"""
		Add layers described in the args.extractor. Layers are separated by a comma.

		:param args: Arguments from commandline
		:return:
		"""
		with tf.name_scope("extractor"):
			extractor_desc = args.extractor.split(',')
			extractor_depth = len(extractor_desc)
			for l in range(extractor_depth):
				layer_name = "extractor_layer{}_{}".format(l, extractor_desc[l])
				specs = extractor_desc[l].split('-')

				# - R-hidden_layer_size: Add a shared dense layer with ReLU activation and specified size. Ex: "R-100"
				if specs[0] == 'R':
					shared_layer = tf.layers.Dense(units=int(specs[1]), activation=tf.nn.relu, name=layer_name)
					for game_node in range(self.NUM_NODES):
						self.latest_shared_layer[game_node] = shared_layer(self.latest_shared_layer[game_node])
				else:
					raise ValueError("Invalid extractor specification '{}'".format(specs))

	def construct_context_pooling(self):
		"""
		Add layers to build context: means and maxs per each public state.

		:return:
		"""
		with tf.name_scope("context_pooling"):
			# scatter nodes by public states
			self.public_states_lists = [[] for _ in range(self.NUM_PUBLIC_STATES)]
			for game_node in range(self.NUM_NODES):
				related_public_state = self._node_to_public_state[game_node]
				self.public_states_lists[related_public_state].append(self.latest_shared_layer[game_node])

			# pooling operations
			public_states_tensors = [None] * self.NUM_PUBLIC_STATES
			public_state_means = [None] * self.NUM_PUBLIC_STATES
			public_state_maxes = [None] * self.NUM_PUBLIC_STATES
			context = [None] * self.NUM_PUBLIC_STATES
			for i, public_state_list in enumerate(self.public_states_lists):
				public_states_tensors[i] = tf.stack(public_state_list, axis=-1, name="nodes_of_public_state{}".format(i))
				with tf.name_scope("public_state{}".format(i)):
					public_state_means[i] = tf.reduce_mean(public_states_tensors[i], axis=-1, name="public_state_mean{}".format(i))
					public_state_maxes[i] = tf.reduce_max(public_states_tensors[i], axis=-1, name="public_state_maxes{}".format(i))
					context[i] = tf.concat(
						[public_state_means[i], public_state_maxes[i]],
						axis=-1,
						name="context{}".format(i)
					)

		with tf.name_scope("concat_context"):
			# concatenate with extractor's outputs to form regressor's input
			for game_node in range(self.NUM_NODES):
				related_public_state = self._node_to_public_state[game_node]
				self.latest_shared_layer[game_node] = tf.concat(
					[self.latest_shared_layer[game_node], context[related_public_state]],
					axis=-1,
					name="features_with_context_of_node{}".format(game_node)
				)

	def construct_value_regressor(self, args):
		"""
		Add layers described in the args.regressor. Layers are separated by a comma.

		:param args: Arguments from commandline
		:return:
		"""
		with tf.name_scope("regressor"):
			regressor_desc = args.regressor.split(',')
			regressor_depth = len(regressor_desc)
			for l in range(regressor_depth):
				layer_name = "regressor_layer{}_{}".format(l, regressor_desc[l])
				specs = regressor_desc[l].split('-')

				# - R-hidden_layer_size: Add a shared dense layer with ReLU activation and specified size. Ex: "R-100"
				if specs[0] == 'R':
					shared_layer = tf.layers.Dense(units=int(specs[1]), activation=tf.nn.relu, name=layer_name)
					for game_node in range(self.NUM_NODES):
						self.latest_shared_layer[game_node] = shared_layer(self.latest_shared_layer[game_node])
				else:
					raise ValueError("Invalid regressor specification '{}'".format(specs))

	def construct(self, args):
		with self.session.graph.as_default():
			# Inputs
			self.features = tf.placeholder(FLOAT_DTYPE, [None, self.NUM_NODES, self.FEATURES_DIM], name="input_features")
			self.targets = tf.placeholder(FLOAT_DTYPE, [None, self.NUM_NODES], name="targets")

			# Computation
			with tf.name_scope("input"):
				self.latest_shared_layer = [
					tf.identity(
						self.features[:, game_node, :],
						name="features_of_node{}".format(game_node)
					)
					for game_node in range(self.NUM_NODES)
				]

			self.construct_feature_extractor(args)
			self.construct_context_pooling()
			self.construct_value_regressor(args)

			# Add final layers to predict nodal equilibrial expected values.
			with tf.name_scope("output"):
				shared_layer = tf.layers.Dense(self.TARGETS_DIM, activation=None)
				self.predictions = [
					tf.identity(
						shared_layer(self.latest_shared_layer[game_node]),
						name="prediction_of_node{}".format(game_node)
					)
					for game_node in range(self.NUM_NODES)
				]
				self.predictions = tf.squeeze(tf.stack(self.predictions, axis=1), name="predictions")

			# Training
			with tf.name_scope("metrics"):
				loss = tf.losses.huber_loss(self.targets, self.predictions, scope="huber_loss")
				with tf.name_scope("mean_squared_error"):
					self.mean_squared_error = tf.reduce_mean(tf.square(self.targets - self.predictions))
				with tf.name_scope("l_infinity_error"):
					self.l_infinity_error = tf.reduce_max(tf.abs(self.targets - self.predictions))
			with tf.name_scope("optimization"):
				global_step = tf.train.create_global_step()
				self.optimizer = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="optimizer")

			# Summaries
			with tf.name_scope("summaries"):
				summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
			self.summaries = {}
			with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
				self.summaries["train"] = [
					tf.contrib.summary.scalar("train/loss", loss),
					tf.contrib.summary.scalar("train/mean_squared_error", self.mean_squared_error),
					tf.contrib.summary.scalar("train/l_infinity_error", self.l_infinity_error)
				]
			with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				for dataset in ["dev", "test"]:
					self.summaries[dataset] = [
						tf.contrib.summary.scalar(dataset + "/loss", loss),
						tf.contrib.summary.scalar(dataset + "/mean_squared_error", self.mean_squared_error),
						tf.contrib.summary.scalar(dataset + "/l_infinity_error", self.l_infinity_error)
					]

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			with summary_writer.as_default():
				tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def train(self, features, targets):
		self.session.run([self.optimizer, self.summaries["train"]], {self.features: features, self.targets: targets})

	def evaluate(self, dataset, features, targets):
		mean_squared_error, l_infinity_error, _ = self.session.run(
			[self.mean_squared_error, self.l_infinity_error, self.summaries[dataset]],
			{self.features: features, self.targets: targets}
		)
		return mean_squared_error, l_infinity_error

	def predict(self, features):
		return self.session.run(self.predictions, {self.features: features})


if __name__ == "__main__":
	import argparse
	import datetime
	import os
	import re

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	if FIXED_RANDOMNESS:
		np.random.seed(SEED_FOR_TESTING)  # Fix random seed

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=10, type=int, help="Batch size.")
	parser.add_argument("--extractor", default="R-{}".format(NeuralNetwork_IIGS3Lvl10.FEATURES_DIM), type=str,
	                    help="Description of the feature extactor architecture.")
	parser.add_argument("--regressor", default="R-{}".format(NeuralNetwork_IIGS3Lvl10.FEATURES_DIM), type=str,
	                    help="Description of the value regressor architecture.")
	parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")

	args = parser.parse_args()

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
	trainset = DatasetFromNPZ("{}/data/IIGS3/80-10-10/IIGS3_1_3_false_true_lvl7_train.npz".format(script_directory))
	devset = DatasetFromNPZ("{}/data/IIGS3/80-10-10/IIGS3_1_3_false_true_lvl7_dev.npz".format(script_directory))
	testset = DatasetFromNPZ("{}/data/IIGS3/80-10-10/IIGS3_1_3_false_true_lvl7_test.npz".format(script_directory))

	# Construct the network
	network = NeuralNetwork_IIGS3Lvl10(threads=args.threads)
	features, targets = trainset.next_batch(args.batch_size)
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
