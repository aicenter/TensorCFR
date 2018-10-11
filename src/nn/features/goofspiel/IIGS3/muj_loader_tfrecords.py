import tensorflow as tf

from tensorflow.python.framework.errors_impl import OutOfRangeError


NUMBER_DATASET_FILES = 1000
BATCH_SIZE = 8
NUM_PARALLEL_CALLS = 2


def _parser(record):
	keys_to_features = {
		'sample_input': tf.FixedLenFeature((7,), tf.float32),
		'sample_target': tf.FixedLenFeature((1,), tf.float32)
	}
	parsed = tf.parse_single_example(record, keys_to_features)
	return parsed["sample_input"], parsed["sample_target"]


if __name__ == '__main__':
	data_path = ['train_{}.tfrecord'.format(x) for x in range(NUMBER_DATASET_FILES)]

	dataset = tf.data.TFRecordDataset(filenames=data_path)
	dataset = dataset.map(_parser, num_parallel_calls=NUM_PARALLEL_CALLS)
	dataset = dataset.batch(BATCH_SIZE)
	iterator = dataset.make_one_shot_iterator()
	features = iterator.get_next()

	# dataset = dataset.apply(
	# 	tf.contrib.data.map_and_batch(parser, 32)
	# )
	# dataset = dataset.prefetch(buffer_size=2)


	with tf.Session() as session:
		while True:
			try:
				# Runs `features` TF op and return a tuple with Numpy arrays
				features_data = session.run(features)
				# Outputs features_data[0] (sample_input) with shape (BATCH_SIZE,7)
				# and features_data[1] (sample_target) with shape (BATCH_SIZE,1)
				print('Sample input ', features_data[0], ', sample target ', features_data[1])
			except OutOfRangeError:
				# Ends the cycle if we run out of data samples
				break