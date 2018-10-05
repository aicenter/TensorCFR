#!/usr/bin/env python3

import os

import numpy as np
import tensorflow as tf

from src.utils.tf_utils import get_default_config_proto, print_tensors


class DatasetFromNPZ:
	def __init__(self, filename, shuffle_batches=True):
		with np.load(filename) as data:
			self._features = data["features"]
			self._targets = data["targets"]

		self._shuffle_batches = shuffle_batches
		self._permutation = np.random.permutation(len(self._features)) if self._shuffle_batches \
			else range(len(self._features))

	@property
	def features(self):
		return self._features

	@property
	def targets(self):
		return self._targets

	def next_batch(self, batch_size):
		batch_size = min(batch_size, len(self._permutation))
		batch_perm, self._permutation = self._permutation[:batch_size], self._permutation[batch_size:]
		return self._features[batch_perm], self._targets[batch_perm] if self._targets is not None else None

	def epoch_finished(self):
		if len(self._permutation) == 0:
			self._permutation = np.random.permutation(len(self._features)) if self._shuffle_batches \
				else range(len(self._features))
			return True
		return False


if __name__ == "__main__":
	import argparse

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=256, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=10, type=int, help="Number of epochs.")
	args = parser.parse_args()

	train_file = "{}/train.npz".format(script_directory)
	train = DatasetFromNPZ(train_file)

	# TODO
	# for i in range(args.epochs):
	# 	while not train.epoch_finished():
	# 		features, targets = train.next_batch(args.batch_size)
	# 		raise NotImplementedError("Show train dataset")  # TODO

	# TODO
	features = train.features
	targets = train.targets

	features_placeholder, targets_placeholder = tf.placeholder(features.dtype, features.shape, name="features"), \
																							tf.placeholder(targets.dtype, targets.shape, name="targets")

	features_dataset, targets_dataset = tf.data.Dataset.from_tensor_slices(features_placeholder), \
																			tf.data.Dataset.from_tensor_slices(targets_placeholder)
	feature_iterator, target_iterator = features_dataset.make_initializable_iterator(), \
																			targets_dataset.make_initializable_iterator()
	features_batch, targets_batch = feature_iterator.get_next(name="features_batch"), \
																	target_iterator.get_next(name="targets_batch")

	with tf.Session(config=get_default_config_proto()) as sess:
		sess.run(feature_iterator.initializer, feed_dict={features_placeholder: features})
		sess.run(target_iterator.initializer, feed_dict={targets_placeholder: targets})
		batch_index = 0
		while True:
			try:
				print("Batch #{}:".format(batch_index))
				print_tensors(sess, [features_batch, targets_batch])
			except tf.errors.OutOfRangeError:
				break
			batch_index += 1
