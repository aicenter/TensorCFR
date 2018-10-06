#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import numpy as np
import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING
from src.nn.data.IIGS3.DatasetFromNPZ import DatasetFromNPZ


class Network:
	NODES = 36
	FEATURES_DIM = 3 * (2 + 2 + 2) + 1    # 6x 1-of-3-hot encodings (3 per hierarchy) + reach probability
	TARGETS_DIM = 1

	def __init__(self, threads, seed=SEED_FOR_TESTING):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                             intra_op_parallelism_threads=threads))

	def construct(self, args):
		with self.session.graph.as_default():
			# Inputs
			self.features = tf.placeholder(tf.float32, [None, self.NODES, self.FEATURES_DIM], name="input_features")
			self.targets = tf.placeholder(tf.int64, [None], name="targets")

			# Computation
			latest_layer = self.features

			# Add layers described in the args.extractor. Layers are separated by a comma.
			extractor_desc = args.extractor.split(',')
			extractor_depth = len(extractor_desc)
			for l in range(extractor_depth):
				layer_name = "extractor_layer{}-{}".format(l, extractor_desc[l])
				specs = extractor_desc[l].split('-')
				if specs[0] == 'R':   # TODO add tf.keras.layers.PReLU
					# - R-hidden_layer_size: Add a shared dense layer with ReLU activation and specified size. Ex: "R-100"
					shared_layer = tf.layers.Dense(units=int(specs[1]), activation=tf.nn.relu, name=layer_name)
					# share layers by mapping from inputs to outputs
					for game_node in range(self.NODES):
						latest_layer[:, game_node, :] = shared_layer(latest_layer[:, game_node, :])

					# TODO remove
					#  - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: "R-100"
					# latest_layer = tf.layers.dense(inputs=latest_layer, units=int(specs[1]), activation=tf.nn.relu,
					#                                name=layer_name)
				else:
					raise ValueError("Invalid extractor specification '{}'".format(specs))

			# TODO modify for `shared_layer`
			# Add layers described in the args.regressor. Layers are separated by a comma.
			# regressor_desc = args.regressor.split(',')
			# regressor_depth = len(regressor_desc)
			# for l in range(regressor_depth):
			# 	layer_name = "regressor_layer{}-{}".format(l, regressor_desc[l])
			# 	specs = regressor_desc[l].split('-')
			# 	if specs[0] == 'R':   # TODO add tf.keras.layers.PReLU
			# 		# - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: "R-100"
			# 		latest_layer = tf.layers.dense(inputs=latest_layer, units=int(specs[1]), activation=tf.nn.relu,
			# 		                               name=layer_name)
			# 	else:
			# 		raise ValueError("Invalid regressor specification '{}'".format(specs))

			# Add final layers to predict nodal equilibrial expected values.
			self.predictions = tf.layers.dense(latest_layer, self.TARGETS_DIM, activation=None, name="output_layer")

			# Training
			# loss = tf.losses.mean_squared_error(self.targets, self.predictions, scope="mse_loss")
			loss = tf.losses.huber_loss(self.targets, self.predictions, scope="huber_loss")
			global_step = tf.train.create_global_step()
			self.training = tf.train.AdamOptimizer().minimize(loss, global_step=global_step, name="training")

			# Summaries
			self.accuracy = tf.reduce_mean(tf.abs(self.targets - self.predictions), tf.float32)   # TODO ask Vilo
			summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
			self.summaries = {}
			with summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
				self.summaries["train"] = [tf.contrib.summary.scalar("train/loss", loss),
				                           tf.contrib.summary.scalar("train/accuracy", self.accuracy)]
			with summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
				for dataset in ["dev", "test"]:
					self.summaries[dataset] = [tf.contrib.summary.scalar(dataset + "/loss", loss),
					                           tf.contrib.summary.scalar(dataset + "/accuracy", self.accuracy)]

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			with summary_writer.as_default():
				tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def train(self, features, targets):
		self.session.run([self.training, self.summaries["train"]], {self.features: features, self.targets: targets})

	def evaluate(self, dataset, features, targets):
		accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]],
		                               {self.features: features, self.targets: targets})
		return accuracy

	def predict(self, features):
		return self.session.run(self.predictions, {self.features: features})


if __name__ == "__main__":
	import argparse
	import datetime
	import os
	import re

	# Fix random seed
	np.random.seed(SEED_FOR_TESTING)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=1, type=int, help="Batch size.")
	parser.add_argument("--extractor", default="R-{}".format(Network.FEATURES_DIM), type=str,
	                    help="Description of the feature extactor architecture.")
	parser.add_argument("--regressor", default="R-{}".format(Network.FEATURES_DIM), type=str,
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
	train_file = "{}/data/IIGS3/train.npz".format(script_directory)
	trainset = DatasetFromNPZ(train_file)
	# TODO devset

	# Construct the network
	network = Network(threads=args.threads)
	network.construct(args)

	# Train
	for epoch in range(args.epochs):
		print("Epoch #{}:".format(epoch))
		while not trainset.epoch_finished():
			print("Batch #{}:".format(trainset.batch_id))
			features, targets = trainset.next_batch(args.batch_size)
			print("Features:\n{}".format(features))
			print("Targets:\n{}".format(targets))
			network.train(features, targets)
		# network.evaluate("dev", mnist.validation.features, mnist.validation.targets)  # TODO devset
