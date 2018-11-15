import numpy as np
import pandas as pd
import tensorflow as tf

from src.commons.constants import FLOAT_DTYPE
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10


class SanityCNN(ConvNet_IIGS6Lvl10):
	NUM_NODES = 3 # 14400
	NUM_ROUNDS = 1 # 3

	PUBLIC_FEATURES_DIM = 3
	INFOSET_FEATURES_DIM = 3 # 6
	NODAL_FEATURES_DIM = 6
	FEATURES_DIM_PER_ROUND = 3 # PUBLIC_FEATURES_DIM + INFOSET_FEATURES_DIM + NODAL_FEATURES_DIM
	REACH_PROB_DIM = 1

	INPUT_FEATURES_DIM = NUM_ROUNDS * FEATURES_DIM_PER_ROUND + REACH_PROB_DIM
	TARGETS_DIM = 1
	NUM_PUBLIC_STATES = 1 # 3 ** NUM_ROUNDS

	def construct_input(self):
		self._one_hot_features_np = np.array([[0,1],[0,0],[1,0]]) # np.array([[0,0,1],[1,0,0]])

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
				name="input_channels_first_NCL"
				# [batch, channels, lengths] == [batch_size, INPUT_FEATURES_DIM, NUM_NODES]
			)
		print("Input constructed")
		self.targets = tf.placeholder(FLOAT_DTYPE, [None, self.NUM_NODES], name="targets")
		print("Targets constructed")
		self.print_operations_count()

	def construct_context_pooling(self):
		"""
		Add layers to build context: means and maxes per each public state.

		:return:
		"""
		self._sizes_of_public_states = [3]

		with tf.variable_scope("context_pooling"):
			groups_by_public_states = tf.split(
				value=self.latest_layer,
				num_or_size_splits= 1, # self._sizes_of_public_states,
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

ACTIVATE_FILE = False

if __name__ == '__main__'  and ACTIVATE_FILE:
	pass