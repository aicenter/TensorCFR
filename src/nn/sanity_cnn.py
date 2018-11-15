import numpy as np
import pandas as pd
import tensorflow as tf

from src.commons.constants import FLOAT_DTYPE
from src.nn.ConvNet_IIGS6Lvl10 import ConvNet_IIGS6Lvl10


class SanityCNN(ConvNet_IIGS6Lvl10):
	NUM_NODES = 3 # 14400
	NUM_ROUNDS = 3

	PUBLIC_FEATURES_DIM = 3
	INFOSET_FEATURES_DIM = 6
	NODAL_FEATURES_DIM = 6
	FEATURES_DIM_PER_ROUND = PUBLIC_FEATURES_DIM + INFOSET_FEATURES_DIM + NODAL_FEATURES_DIM
	REACH_PROB_DIM = 1

	INPUT_FEATURES_DIM = NUM_ROUNDS * FEATURES_DIM_PER_ROUND + REACH_PROB_DIM
	TARGETS_DIM = 1
	NUM_PUBLIC_STATES = 1 # 3 ** NUM_ROUNDS

	def construct_input(self):
		self._one_hot_features_np = np.array([[0,0,1],[1,0,0]])

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

if __name__ == '__main__':
	pass