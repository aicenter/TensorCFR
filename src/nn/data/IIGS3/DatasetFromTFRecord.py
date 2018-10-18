#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

from src.utils.other_utils import get_files_in_directory_recursively


class DatasetFromTFRecord:
	def __init__(self,
				 batch_size=1,
				 dataset_files=list(),
				 feature_input_size=1,
				 feature_target_size=1,
				 number_of_epochs=1,
				 number_parallel_calls=None,
				 shuffle_batches=True,
				 shuffle_batches_buffer_size=100000
				 ):
		self._batch_id = 0
		self._batch_size = batch_size
		self._dataset_files = dataset_files
		self._feature_input_size = feature_input_size
		self._feature_target_size = feature_target_size
		self._features_op = None # TensorFlow operation
		self._number_of_epochs = number_of_epochs
		self._number_parallel_calls = number_parallel_calls
		self._variable_scope = 'DatasetFromTFRecord'
		self._shuffle_batches = shuffle_batches
		self._shuffle_batches_buffer_size = shuffle_batches_buffer_size

		self._init()

	@property
	def batch_id(self):
		return self._batch_id

	@property
	def features(self):
		return None

	@property
	def targets(self):
		return None

	def _init(self):
		if self._features_op is None:
			with tf.variable_scope(self._variable_scope):
				dataset = tf.data.TFRecordDataset(filenames=self._dataset_files)
				dataset = dataset.repeat(self._number_of_epochs)
				dataset = dataset.map(
					lambda tfrecord_element: self._parser(tfrecord_element),
					num_parallel_calls=self._number_parallel_calls)
				if self._shuffle_batches:
					dataset = dataset.shuffle(buffer_size=self._shuffle_batches_buffer_size)
				dataset = dataset.batch(self._batch_size)
				iterator = dataset.make_one_shot_iterator()
				self._features_op = iterator.get_next()

	def _parser(self, tfrecord_element):
		keys_to_features = {
			'dataset_sample_input': tf.FixedLenFeature((self._feature_input_size,), tf.float32),
			'dataset_sample_target': tf.FixedLenFeature((self._feature_target_size,), tf.float32)
		}
		parsed = tf.parse_single_example(tfrecord_element, keys_to_features)
		return parsed["dataset_sample_input"], parsed["dataset_sample_target"]

	def next_batch(self):
		self._batch_id += 1
		return self._features_op

	def epoch_finished(self):
		pass

if __name__ == "__main__":
	import argparse

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=3, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
	parser.add_argument("--feature_input_size", default=36, type=int, help="Size of the .")
	parser.add_argument("--feature_target_size", default=36, type=int, help="Number of epochs.")
	args = parser.parse_args()

	dataset_files = get_files_in_directory_recursively(rootdir='/home/ruda/Documents/Projects/tensorcfr/TensorCFR/src/nn/features/goofspiel/IIGS3/tfrecord_dataset_IIGS3_1_3_false_true_lvl7')
	# dataset_files = ["dataset_{}.tfrecord".format(x) for x in range(34)]

	dataset = DatasetFromTFRecord(dataset_files=dataset_files, feature_input_size=36, feature_target_size=36)

	# for epoch in range(args.epochs):
	# 	print("Epoch #{}:".format(epoch))
	# 	while not train.epoch_finished():
	# 		print("Batch #{}:".format(train.batch_id))
	# 		features, targets = train.next_batch(args.batch_size)
	# 		print("Features:\n{}".format(features))
	# 		print("Targets:\n{}".format(targets))
	with tf.Session() as sess:
		while True:
			try:
				feature_input, feature_target = sess.run(dataset.next_batch())
				print(feature_input, feature_target)
			except tf.errors.OutOfRangeError:
				# Ends the cycle if we run out of data samples
				break


