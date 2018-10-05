#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import numpy as np
import tensorflow as tf


class Network:
	WIDTH = 28
	HEIGHT = 28
	LABELS = 10

	def __init__(self, threads, seed=42):
		# Create an empty graph and a session
		graph = tf.Graph()
		graph.seed = seed
		self.session = tf.Session(graph=graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
		                                                             intra_op_parallelism_threads=threads))

	def construct(self, args, batches_per_epoch, decay_rate):
		with self.session.graph.as_default():
			# Inputs
			self.features = tf.placeholder(tf.float32, [None, self.HEIGHT, self.WIDTH, 1], name="features")
			self.targets = tf.placeholder(tf.int64, [None], name="targets")
			self.is_training = tf.placeholder(tf.bool, [], name="is_training")

			# Computation
			latest_layer = self.features
			# Add layers described in the args.cnn. Layers are separated by a comma and can be:
			cnn_desc = args.cnn.split(',')
			depth = len(cnn_desc)
			for l in range(depth):
				layer_name = "layer{}-{}".format(l, cnn_desc[l])
				specs = cnn_desc[l].split('-')
				if specs[0] == 'M':
					# - M-kernel_size-stride: Add max pooling with specified size and stride. Example: M-3-2
					latest_layer = tf.layers.max_pooling2d(inputs=latest_layer, pool_size=int(specs[1]), strides=int(specs[2]),
					                                       name=layer_name)
				if specs[0] == 'F':
					# - F: Flatten inputs
					latest_layer = tf.layers.flatten(inputs=latest_layer, name=layer_name)
				if specs[0] == 'R':
					# - R-hidden_layer_size: Add a dense layer with ReLU activation and specified size. Ex: R-100
					latest_layer = tf.layers.dense(inputs=latest_layer, units=int(specs[1]), activation=tf.nn.relu,
					                               name=layer_name)
				if specs[0] == 'CB':
					# - CB-filters-kernel_size-stride-padding: Add a convolutional layer with BatchNorm
					#   and ReLU activation and specified number of filters, kernel size, stride and padding.
					#   Example: CB-10-3-1-same
					# To correctly implement BatchNorm:
					# - The convolutional layer should not use any activation and no biases.
					conv_layer = tf.layers.conv2d(inputs=latest_layer, filters=int(specs[1]), kernel_size=int(specs[2]),
					                              strides=int(specs[3]), padding=specs[4], activation=None, use_bias=False)
					# - The output of the convolutional layer is passed to batch_normalization layer, which
					#   should specify `training=True` during training and `training=False` during inference.
					batchnorm_layer = tf.layers.batch_normalization(inputs=conv_layer, training=self.is_training)
					# - The output of the batch_normalization layer is passed through tf.nn.relu.
					latest_layer = tf.nn.relu(batchnorm_layer, name=layer_name)

				# # Implement dropout on the hidden layer using tf.layers.dropout,
				# # with using dropout date of args.dropout. The dropout must be active only
				# # during training -- use `self.is_training` placeholder to control the
				# # `training` argument of tf.layers.dropout. Store the result to `hidden_layer_dropout`.
				# hidden_layer_dropout = tf.layers.dropout(hidden_layer, rate=args.dropout, training=self.is_training,
				# 																				 name="hidden_layer_dropout")
				# # output_layer = tf.layers.dense(hidden_layer_dropout, self.LABELS, activation=None, name="output_layer")

			# Store result in `features`.
			features = latest_layer

			output_layer = tf.layers.dense(features, self.LABELS, activation=None, name="output_layer")
			self.predictions = tf.argmax(output_layer, axis=1)

			# Training
			loss = tf.losses.sparse_softmax_cross_entropy(self.targets, output_layer, scope="loss")
			global_step = tf.train.create_global_step()
			learning_rate = tf.train.exponential_decay(args.learning_rate, global_step, batches_per_epoch, decay_rate,
			                                           staircase=True)
			# - You need to update the moving averages of mean and variance in the batch normalization
			#   layer during each training batch. Such update operations can be obtained using
			#   `tf.get_collection(tf.GraphKeys.UPDATE_OPS)` and utilized either directly in `session.run`,
			#   or (preferably) attached to `self.train` using `tf.control_dependencies`.
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			with tf.control_dependencies(update_ops):
				self.training = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step, name="training")

			# Summaries
			self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.targets, self.predictions), tf.float32))
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

	def train(self, images, labels):
		self.session.run([self.training, self.summaries["train"]], {self.features   : images, self.targets: labels,
		                                                            self.is_training: True})

	def evaluate(self, dataset, images, labels):
		accuracy, _ = self.session.run([self.accuracy, self.summaries[dataset]], {self.features   : images, self.targets: labels,
		                                                                          self.is_training: False})
		return accuracy

	def predict(self, images):
		return self.session.run(self.predictions, {self.features: images, self.is_training: False})


if __name__ == "__main__":
	import argparse
	import datetime
	import os
	import re

	# Fix random seed
	np.random.seed(42)

	# Parse arguments
	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=50, type=int, help="Batch size.")
	parser.add_argument("--dropout", default=0.6, type=float, help="Dropout rate.")
	parser.add_argument("--cnn", default=None, type=str, help="Description of the CNN architecture.")
	parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
	parser.add_argument("--threads", default=1, type=int, help="Maximum number of threads to use.")
	parser.add_argument("--learning_rate", default=0.001, type=float, help="Initial learning rate.")
	parser.add_argument("--learning_rate_final", default=None, type=float, help="Final learning rate.")

	args = parser.parse_args()

	# Create logdir name
	args.logdir = "logs/{}-{}-{}".format(
		os.path.basename(__file__),
		datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S"),
		",".join(("{}={}".format(re.sub("(.)[^_]*_?", r"\1", key), value) for key, value in sorted(vars(args).items())))
	)
	if not os.path.exists("logs"): os.mkdir("logs")  # TF 1.6 will do this by itself

	# Load the data
	from tensorflow.examples.tutorials import mnist

	mnist = mnist.input_data.read_data_sets("mnist-gan", reshape=False, seed=42,
	                                        source_url="https://ufal.mff.cuni.cz/~straka/courses/npfl114/1718/mnist-gan/")

	# Construct the network
	# set up decay rate
	if args.learning_rate_final is not None:
		decay_rate = np.power(args.learning_rate_final / args.learning_rate, 1 / (args.epochs - 1))
	else:
		decay_rate = 1.0
	batches_per_epoch = mnist.train.num_examples // args.batch_size

	network = Network(threads=args.threads)
	network.construct(args, batches_per_epoch, decay_rate)

	# Train
	for i in range(args.epochs):
		j = 0
		while mnist.train.epochs_completed == i:
			print("Epoch #{} \t Batch #{}".format(i, j))
			j += 1
			images, labels = mnist.train.next_batch(args.batch_size)
			network.train(images, labels)

		network.evaluate("dev", mnist.validation.features, mnist.validation.targets)

	# Compute test_labels, as numbers 0-9, corresponding to mnist.test.features
	test_labels = network.predict(mnist.test.features)
	test_filename = "{}/mnist_competition_test.txt".format(args.logdir)
	with open(test_filename, "w") as test_file:
		for label in test_labels:
			print(label, file=test_file)
