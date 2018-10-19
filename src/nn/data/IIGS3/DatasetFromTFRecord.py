#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

from src.commons.constants import PROJECT_ROOT
from src.utils.other_utils import get_files_in_directory_recursively


class DatasetFromTFRecord:
	def __init__(self, batch_size=1, dataset_files=list(), feature_input_size=1, feature_target_size=1,
	             number_of_epochs=1, number_parallel_calls=None, variable_scope_name='DatasetFromTFRecord',
	             shuffle_batches=True, shuffle_batches_buffer_size=100000):
		self.iterator = None

		self.epoch_finished = None

		self._batch_id = 0
		self._batch_size = batch_size
		self._dataset_files = dataset_files
		self._feature_input_size = feature_input_size
		self._feature_target_size = feature_target_size
		self._features_op = None  # TensorFlow operation
		self._number_of_epochs = number_of_epochs
		self._number_parallel_calls = number_parallel_calls
		self._variable_scope_name = variable_scope_name
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
			with tf.variable_scope(self._variable_scope_name):
				dataset = tf.data.TFRecordDataset(filenames=self._dataset_files)
				dataset = dataset.repeat(self._number_of_epochs)
				dataset = dataset.map(
					lambda tfrecord_element: self._parser(tfrecord_element),
					num_parallel_calls=self._number_parallel_calls)
				if self._shuffle_batches:
					dataset = dataset.shuffle(buffer_size=self._shuffle_batches_buffer_size)
				dataset = dataset.batch(self._batch_size)
				self.iterator = dataset.make_initializable_iterator()
				self._features_op = self.iterator.get_next()

	def _parser(self, tfrecord_element):
		keys_to_features = {
			'dataset_sample_input' : tf.FixedLenFeature((self._feature_input_size,), tf.float32),
			'dataset_sample_target': tf.FixedLenFeature((self._feature_target_size,), tf.float32)
		}
		parsed = tf.parse_single_example(tfrecord_element, keys_to_features)
		return parsed["dataset_sample_input"], parsed["dataset_sample_target"]

	def next_batch(self, session):
		if self.epoch_finished is None or self.epoch_finished is True:
			session.run(self.iterator.initializer)
			self.epoch_finished = False
			self._batch_id = 0

		while True:
			self._batch_id += 1
			try:
				feature_input, feature_target = session.run(self._features_op)
				if feature_target.shape[0] != self._batch_size:
					self.epoch_finished = True
					break
			except tf.errors.OutOfRangeError:
				self.epoch_finished = True
				break
			yield (feature_input, feature_target)

	def epoch_finished(self):
		return self.epoch_finished


if __name__ == "__main__":
	import argparse

	np.set_printoptions(edgeitems=20, suppress=True, linewidth=200)
	script_directory = os.path.dirname(os.path.abspath(__file__))
	np.random.seed(42)

	parser = argparse.ArgumentParser()
	parser.add_argument("--batch_size", default=8, type=int, help="Batch size.")
	parser.add_argument("--epochs", default=2, type=int, help="Number of epochs.")
	parser.add_argument("--feature_input_size", default=36, type=int, help="Size of the .")
	parser.add_argument("--feature_target_size", default=36, type=int, help="Number of epochs.")
	args = parser.parse_args()

	dataset_files = get_files_in_directory_recursively(
		rootdir=os.path.join(PROJECT_ROOT, 'src', 'nn', 'features', 'goofspiel', 'IIGS3',
		                     'tfrecord_dataset_IIGS3_1_3_false_true_lvl7')
	)

	trainset_ratio = 0.8
	devset_ratio = 0.1
	dataset_size = len(dataset_files)
	split_train = int(trainset_ratio * dataset_size)
	split_dev = int((trainset_ratio + devset_ratio) * dataset_size)

	training_set_dataset_files = dataset_files[0:split_train]
	dev_set_dataset_files = dataset_files[split_train:]

	train_dataset = DatasetFromTFRecord(
		batch_size=args.batch_size,  # 8
		dataset_files=training_set_dataset_files,
		feature_input_size=args.feature_input_size,  # 36
		feature_target_size=args.feature_input_size,  # 36
		variable_scope_name='train_dataset'
	)
	dev_dataset = DatasetFromTFRecord(
		batch_size=args.batch_size,  # 8
		dataset_files=dev_set_dataset_files,
		feature_input_size=args.feature_input_size,  # 36
		feature_target_size=args.feature_input_size,  # 36
		variable_scope_name='test_dataset'
	)

	with tf.Session() as sess:
		for epoch in range(args.epochs):
			print('Epoch', epoch)
			print('\tTrain set:')
			for sample in train_dataset.next_batch(sess):
				feature_input, feature_target = sample
				print(feature_input, feature_target)
			print('\tDev (Validation) set:')
			for sample in dev_dataset.next_batch(sess):
				feature_input, feature_target = sample
				print(feature_input, feature_target)
