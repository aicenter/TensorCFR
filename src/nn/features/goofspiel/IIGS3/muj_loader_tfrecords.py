import tensorflow as tf
import numpy as np


# with tf.Session() as sess:
# 	feature = {'train/input': tf.FixedLenFeature([], tf.float32),
# 			   'train/target': tf.FixedLenFeature([], tf.float32)}
# 	# Create a list of filenames and pass it to a queue
# 	filename_queue = tf.train.string_input_producer(data_path, num_epochs=1)
# 	# Define a reader and read the next record
# 	reader = tf.TFRecordReader()
# 	_, serialized_example = reader.read(filename_queue)
# 	# Decode the record read by the reader
# 	features = tf.parse_single_example(serialized_example, features=feature)
# 	# Convert the image data from string back to the numbers
# 	image = tf.decode_raw(features['train/input'], tf.float32)
#
# 	# # Cast label data into int32
# 	# label = tf.cast(features['train/target'], tf.int32)
# 	# # Reshape image data into the original shape
# 	# image = tf.reshape(image, [224, 224, 3])
# 	#
# 	# # Any preprocessing here ...
#
# 	# Creates batches by randomly shuffling tensors
# 	# images, labels = tf.train.shuffle_batch([image, label], batch_size=10, capacity=30, num_threads=1,
# 	# 										min_after_dequeue=10)
from tensorflow.python.framework.errors_impl import OutOfRangeError


def parser(record):
	keys_to_features = {
		'sample_input': tf.FixedLenFeature((7,), tf.float32),
		'sample_target': tf.FixedLenFeature((1,), tf.float32)
	}
	parsed = tf.parse_single_example(record, keys_to_features)
	return parsed["sample_input"], parsed["sample_target"]

if __name__ == '__main__':
	data_path = ['train_{}.tfrecord'.format(x) for x in range(1)]  # address to save the hdf5 file

	dataset = tf.data.TFRecordDataset(filenames=data_path)
	# dataset = dataset.apply(
	# 	tf.contrib.data.map_and_batch(parser, 32)
	# )
	# dataset = dataset.prefetch(buffer_size=2)

	dataset = dataset.map(parser)
	iterator = dataset.make_one_shot_iterator()
	features = iterator.get_next()

	with tf.Session() as session:
		while True:
			try:
				features_data = session.run(features)
				print('Sample input ', features_data[0], ', sample target ', features_data[1])
			except OutOfRangeError:
				break