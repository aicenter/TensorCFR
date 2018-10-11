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

def parser(record):
	keys_to_features = {
		#"train/input": tf.FixedLenFeature((), tf.float32),
		"target": tf.FixedLenFeature((), tf.float32)
	}
	parsed = tf.parse_single_example(record, keys_to_features)

	#sample_input = tf.cast(parsed["train/input"], tf.float32)
	# label = tf.cast(parsed["train/target"], tf.float32)
	# return sample_input
	return parsed["target"]# , parsed["train/target"]
	#return parsed

if __name__ == '__main__':
	data_path = ['train_{}.tfrecord'.format(x) for x in range(1)]  # address to save the hdf5 file

	dataset = tf.data.TFRecordDataset(filenames=data_path)
	# dataset = dataset.apply(
	# 	tf.contrib.data.map_and_batch(parser, 32)
	# )
	# dataset = dataset.prefetch(buffer_size=2)
	print("TFRecord")
	print(type(dataset))
	print(dir(dataset))
	print(dataset)
	dataset = dataset.map(parser)

	iterator = dataset.make_one_shot_iterator()
	# sample_input, sample_target = iterator.get_next()
	features = iterator.get_next()

	# print(features[0])
	# print(features[0].eval())
	# print(type(features[0]))
	# print(dir(features[0]))

	with tf.Session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		# print(sess.run(features[0]))
		print(sess.run(features))
		print(sess.run(features))