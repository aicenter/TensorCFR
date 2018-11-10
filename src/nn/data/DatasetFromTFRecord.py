#!/usr/bin/env python3
import tensorflow as tf


class DatasetFromTFRecord:
	def __init__(self, batch_size=1, dataset_files=list(), sample_length=1, number_of_epochs=1,
	             number_parallel_calls=None, variable_scope_name='DatasetFromTFRecord', shuffle_batches=True,
	             shuffle_batches_buffer_size=512):
		self.iterator = None
		self.epoch_finished = None

		self._batch_id = 0
		self._batch_size = batch_size
		self._dataset_files = dataset_files
		self._input_length = sample_length
		self._target_length = sample_length
		self._features_op = None  # TensorFlow operation
		self._number_of_epochs = number_of_epochs
		self._number_parallel_calls = number_parallel_calls
		self._variable_scope_name = variable_scope_name
		self._shuffle_batches = shuffle_batches
		self._shuffle_batches_buffer_size = shuffle_batches_buffer_size

		if self._features_op is None:
			with tf.variable_scope(self._variable_scope_name):
				dataset = tf.data.TFRecordDataset(filenames=self._dataset_files)
				dataset = dataset.repeat(self._number_of_epochs)
				dataset = dataset.map(
					lambda tfrecord_element: self._parser(tfrecord_element),
					num_parallel_calls=self._number_parallel_calls
				)
				if self._shuffle_batches:
					dataset = dataset.shuffle(buffer_size=self._shuffle_batches_buffer_size)
				dataset = dataset.batch(self._batch_size)
				self.iterator = dataset.make_initializable_iterator()
				self._features_op = self.iterator.get_next()

	@property
	def batch_id(self):
		return self._batch_id

	@property
	def features(self):
		raise NotImplementedError

	@property
	def targets(self):
		raise NotImplementedError

	def _parser(self, tfrecord_element):
		keys_to_features = {
			'dataset_sample_input' : tf.FixedLenFeature((self._input_length,), tf.float32),
			'dataset_sample_target': tf.FixedLenFeature((self._target_length,), tf.float32)
		}
		parsed = tf.parse_single_example(tfrecord_element, keys_to_features)
		return parsed["dataset_sample_input"], parsed["dataset_sample_target"]

	def next_batch(self, session):
		if self.epoch_finished is None or self.epoch_finished is True:
			# TODO debug OOM: Runner_*_TFRecords, command-line args --train_ratio .6, --dev_ratio .2
			# run_options = tf.RunOptions(tf.RunOptions.report_tensor_allocations_upon_oom)
			session.run(self.iterator.initializer)
			self.epoch_finished = False
			self._batch_id = 0

		while True:
			self._batch_id += 1
			try:
				# TODO debug OOM: Runner_*_TFRecords, command-line args --train_ratio .6, --dev_ratio .2
				# run_options = tf.RunOptions(tf.RunOptions.report_tensor_allocations_upon_oom)
				feature_input, feature_target = session.run(self._features_op)
				if feature_target.shape[0] != self._batch_size:
					self.epoch_finished = True
					yield (feature_input, feature_target)
					break
			except tf.errors.OutOfRangeError:
				self.epoch_finished = True
				break
			yield (feature_input, feature_target)

	def epoch_finished(self):
		return self.epoch_finished
