#!/usr/bin/env python3

# from https://www.tensorflow.org/guide/datasets#consuming_numpy_arrays

import os

import numpy as np
import tensorflow as tf

from src.nn.features.goofspiel.prepare_datasets_IIGS3_1_3_false_true_lvl7 import prepare_dataset
from src.utils.tf_utils import get_default_config_proto, print_tensors

FEATURES_BASENAME = "IIGS3_1_3_false_true_lvl7"

if __name__ == '__main__':
	np.set_printoptions(edgeitems=20, suppress=True, linewidth=150)

	script_directory = os.path.dirname(os.path.abspath(__file__))
	npz_filename = "{}/{}_numpy_dataset.npz".format(script_directory, FEATURES_BASENAME)
	if not os.path.isfile(npz_filename):
		npz_created = prepare_dataset()
	else:
		npz_created = True

	if npz_created:
		with np.load(npz_filename) as data:
			features = data["features"]
			targets = data["targets"]

		# Assume that each row of `features` corresponds to the same row as `labels`.
		assert features.shape[0] == targets.shape[0]

		features_placeholder, targets_placeholder = tf.placeholder(features.dtype, features.shape, name="features"), \
																								tf.placeholder(targets.dtype, targets.shape, name="targets")

		features_dataset, targets_dataset = tf.data.Dataset.from_tensor_slices(features_placeholder), \
																				tf.data.Dataset.from_tensor_slices(targets_placeholder)
		feature_iterator, target_iterator = features_dataset.make_initializable_iterator(), \
																				targets_dataset.make_initializable_iterator()
		features_batch, targets_batch = feature_iterator.get_next(name="features_batch"), \
																		target_iterator.get_next(name="targets_batch")
		batch_means = tf.reduce_mean(features_batch, axis=0)

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

			sess.run(feature_iterator.initializer, feed_dict={features_placeholder: features})
			batch_index = 0
			while True:
				try:
					print("Batch #{}:".format(batch_index))
					print_tensors(sess, [batch_means])
				except tf.errors.OutOfRangeError:
					break
				batch_index += 1
