#!/usr/bin/env python3

# taken from https://github.com/ufal/npfl114/blob/3b35b431be3c84c2f2d51a4e2353d65cd30ee8fe/labs/04/mnist_competition.py
import numpy as np
import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING, FLOAT_DTYPE
from src.nn.features.goofspiel.IIGS6.node_to_public_states_IIGS6_1_6_false_true_lvl10 import get_node_to_public_state, \
	get_sizes_of_public_states
from src.nn.features.goofspiel.IIGS6.one_hot_rounds_cards_IIGS6_1_6_false_true_lvl10 import \
	get_1hot_round_card_features_np
from src.utils.tf_utils import count_graph_operations

FIXED_RANDOMNESS = False


class ConvNet_IIGS6Lvl10:
	NUM_NODES = 14400
	NUM_ROUNDS = 3

	PUBLIC_FEATURES_DIM = 3
	INFOSET_FEATURES_DIM = 6
	NODAL_FEATURES_DIM = 6
	FEATURES_DIM_PER_ROUND = PUBLIC_FEATURES_DIM + INFOSET_FEATURES_DIM + NODAL_FEATURES_DIM
	REACH_PROB_DIM = 1

	INPUT_FEATURES_DIM = NUM_ROUNDS * FEATURES_DIM_PER_ROUND + REACH_PROB_DIM
	TARGETS_DIM = 1
	NUM_PUBLIC_STATES = 3 ** NUM_ROUNDS

	def __init__(self, threads, seed=SEED_FOR_TESTING, verbose=True):
		# Create an empty graph and a session
		self.graph = tf.Graph()
		if FIXED_RANDOMNESS:
			self.graph.seed = seed
			self.session = tf.Session(graph=self.graph, config=tf.ConfigProto(inter_op_parallelism_threads=threads,
			                                                                  intra_op_parallelism_threads=threads))
		else:
			self.session = tf.Session(graph=self.graph)

		self._node_to_public_state = get_node_to_public_state()
		self._sizes_of_public_states = get_sizes_of_public_states()
		self._one_hot_features_np = get_1hot_round_card_features_np()

		if verbose:
			print("node_to_public_state:\n{}".format(self._node_to_public_state))
			print("sizes_of_public_states:\n{}".format(self._sizes_of_public_states))
			print("one_hot_features:\n{}".format(self._one_hot_features_np))

	def construct_input(self):
		with tf.variable_scope("input"):
			self.input_reaches = tf.placeholder(
				FLOAT_DTYPE,
				[None, self.NUM_NODES],
				name="input_reaches"
			)
			self.expanded_reaches = tf.expand_dims(
				self.input_reaches,
				axis=-1,
				name="expanded_reaches"
			)

			self._one_hot_features_tf = tf.constant(
				self._one_hot_features_np,
				dtype=FLOAT_DTYPE,
				name="one_hot_features"
			)
			print("one_hot_features.shape: {}".format(self._one_hot_features_tf.shape))
			self.tiled_features = tf.tile(
				tf.expand_dims(self._one_hot_features_tf, axis=0),
				multiples=[tf.shape(self.input_reaches)[0], 1, 1],
				name="tiled_1hot_features"
			)

			self.full_input = tf.concat(
				[self.tiled_features, self.expanded_reaches],
				axis=-1,
				name="full_input"
			)
			self.latest_layer = tf.transpose(  # channels first for GPU computation
				self.full_input,
				perm=[0, 2, 1],
				name="input_channels_first_NCL"  # [batch, channels, lengths] == [batch_size, INPUT_FEATURES_DIM, NUM_NODES]
			)
		print("Input constructed")
		self.targets = tf.placeholder(FLOAT_DTYPE, [None, self.NUM_NODES], name="targets")
		print("Targets constructed")
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
				# - C-hidden_layer_size: 1D convolutional with ReLU activation and specified output size (channels). Ex: "C-100"
				if specs[0] == 'C':
					self.latest_layer = tf.layers.conv1d(
						inputs=self.latest_layer,
						filters=int(specs[1]),
						kernel_size=1,
						activation=tf.nn.relu,
						data_format="channels_first",
						name=layer_name
					)
					print("{} constructed".format(layer_name))
				else:
					raise ValueError("Invalid extractor specification '{}'".format(specs))

	def construct_context_pooling(self):
		"""
		Add layers to build context: means and maxes per each public state.

		:return:
		"""
		with tf.variable_scope("context_pooling"):
			groups_by_public_states = tf.split(
				value=self.latest_layer,
				num_or_size_splits=self._sizes_of_public_states,
				axis=-1,     # NCL
				name="split_by_public_states"
			)

			# pooling operations
			public_state_means = [None] * self.NUM_PUBLIC_STATES
			public_state_maxes = [None] * self.NUM_PUBLIC_STATES
			public_state_contexts = [None] * self.NUM_PUBLIC_STATES
			tiled_contexts = [None] * self.NUM_PUBLIC_STATES
			for i, group in enumerate(groups_by_public_states):
				with tf.variable_scope("public_state{}".format(i)):
					public_state_means[i] = tf.reduce_mean(group, axis=-1, keepdims=True, name="mean{}".format(i))
					public_state_maxes[i] = tf.reduce_max(group, axis=-1, keepdims=True, name="max{}".format(i))
					public_state_contexts[i] = tf.concat(
						[public_state_means[i], public_state_maxes[i]],
						axis=1,
						name="context{}".format(i)
					)
					tiled_contexts[i] = tf.tile(
						public_state_contexts[i],
						multiples=[1, 1, self._sizes_of_public_states[i]],
						name="tiled_context{}".format(i)
					)

		# concatenate with extractor's outputs to form regressor's input
		with tf.variable_scope("concatenate"):
			full_context = tf.concat(
				tiled_contexts,
				axis=-1,
				name="full_context"
			)
			self.latest_layer = tf.concat(
				[self.latest_layer, full_context],
				axis=1,
				name="features_with_contexts"
			)

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
						activation=tf.nn.relu,
						data_format="channels_first",
						name=layer_name
					)
					print("{} constructed".format(layer_name))
				else:
					raise ValueError("Invalid regressor specification '{}'".format(specs))

	def construct_predictions(self):
		with tf.variable_scope("predictions"):
			self.predictions = tf.layers.conv1d(
				inputs=self.latest_layer,
				filters=self.TARGETS_DIM,
				kernel_size=1,
				activation=None,
				data_format="channels_first",
				name="conv1d_regression"
			)
			self.predictions = tf.squeeze(self.predictions, name="predictions")
		print("predictions constructed")
		self.print_operations_count()

	def construct_loss(self):
		with tf.variable_scope("loss"):
			self.loss = self.huber_loss
			print("loss constructed")
			self.print_operations_count()

	def construct_training(self):
		with tf.variable_scope("metrics"):
			with tf.variable_scope("huber_loss"):
				self.huber_loss = tf.losses.huber_loss(self.targets, self.predictions, scope="huber_loss")
			with tf.variable_scope("mean_squared_error"):
				self.mean_squared_error = tf.reduce_mean(tf.squared_difference(self.targets, self.predictions))
			with tf.variable_scope("l_infinity_error"):
				self.l_infinity_error = tf.norm(self.targets - self.predictions, ord=np.inf)
			self.construct_loss()
		print("Metrics constructed")
		self.print_operations_count()
		with tf.variable_scope("optimization"):
			global_step = tf.train.create_global_step()
			print("global_step constructed")
			self.print_operations_count()
			optimizer = tf.train.AdamOptimizer()
			print("optimizer constructed")
			self.print_operations_count()
			self.loss_minimizer = optimizer.minimize(self.loss, global_step=global_step, name="loss_minimizer")
			print("loss_minimizer constructed")
			self.print_operations_count()
		print("Optimization constructed")
		self.print_operations_count()

	def construct_summaries(self, args):
		with tf.variable_scope("summaries"):
			self.summary_writer = tf.contrib.summary.create_file_writer(args.logdir, flush_millis=10 * 1000)
		self.summaries = {}
		with self.summary_writer.as_default(), tf.contrib.summary.record_summaries_every_n_global_steps(100):
			self.summaries["train"] = [
				tf.contrib.summary.scalar("train/loss", self.loss),
				tf.contrib.summary.scalar("train/mean_squared_error", self.mean_squared_error),
				tf.contrib.summary.scalar("train/l_infinity_error", self.l_infinity_error)
			]
		print("Summaries[train] constructed")
		self.print_operations_count()
		with self.summary_writer.as_default(), tf.contrib.summary.always_record_summaries():
			for dataset in ["dev", "test"]:
				self.summaries[dataset] = [
					tf.contrib.summary.scalar(dataset + "/loss", self.loss),
					tf.contrib.summary.scalar(dataset + "/mean_squared_error", self.mean_squared_error),
					tf.contrib.summary.scalar(dataset + "/l_infinity_error", self.l_infinity_error)
				]
		print("Summaries[dev/test] constructed")
		self.print_operations_count()

	def construct(self, args):
		with self.session.graph.as_default():
			# Inputs
			self.construct_input()

			# Computation
			self.construct_feature_extractor(args)
			print("Extractor constructed")
			self.print_operations_count()
			self.construct_context_pooling()
			print("Context pooling constructed")
			self.print_operations_count()
			self.construct_value_regressor(args)
			print("Regressor constructed")
			self.print_operations_count()

			# Add final layers to predict nodal equilibrial expected values.
			self.construct_predictions()

			# Training
			self.construct_training()

			# Summaries
			self.construct_summaries(args)

			# Initialize variables
			self.session.run(tf.global_variables_initializer())
			with self.summary_writer.as_default():
				tf.contrib.summary.initialize(session=self.session, graph=self.session.graph)

	def train(self, features, targets):
		self.session.run([self.loss_minimizer, self.summaries["train"]],
		                 {self.input_reaches: features, self.targets: targets})

	def evaluate(self, dataset, features, targets):
		mean_squared_error, l_infinity_error, _ = self.session.run(
			[self.mean_squared_error, self.l_infinity_error, self.summaries[dataset]],
			{self.input_reaches: features, self.targets: targets}
		)
		return mean_squared_error, l_infinity_error

	def predict(self, features):
		return self.session.run(self.predictions, {self.input_reaches: features})

	def print_operations_count(self):
		print("--> Total size of computation graph: {} operations".format(count_graph_operations(self.graph)))
