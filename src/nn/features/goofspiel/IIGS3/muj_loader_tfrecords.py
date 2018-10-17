import tensorflow as tf

from tensorflow.python.framework.errors_impl import OutOfRangeError


NUMBER_DATASET_FILES = 35
BATCH_SIZE = 8
NUM_PARALLEL_CALLS = 2


def _parser(record):
	keys_to_features = {
		'sample_input': tf.FixedLenFeature((18,), tf.float32),
		'sample_target': tf.FixedLenFeature((1,), tf.float32)
	}
	parsed = tf.parse_single_example(record, keys_to_features)


	return parsed["sample_input"], parsed["sample_target"]


if __name__ == '__main__':
	data_path = ['dataset_{}.tfrecord'.format(x) for x in range(NUMBER_DATASET_FILES)]


	with tf.variable_scope('dataset_loading'):
		dataset = tf.data.TFRecordDataset(filenames=data_path)
		dataset = dataset.repeat(2) # number of epochs
		dataset = dataset.map(_parser, num_parallel_calls=NUM_PARALLEL_CALLS)
		dataset = dataset.batch(BATCH_SIZE)
		iterator = dataset.make_one_shot_iterator()
		features = iterator.get_next()

	# dataset = dataset.apply(
	# 	tf.contrib.data.map_and_batch(parser, 32)
	# )
	# dataset = dataset.prefetch(buffer_size=2)

	m_nn_input = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 18))
	m_nn_target = tf.placeholder(tf.float32, shape=(BATCH_SIZE, 18))

	with tf.Session() as session:
		writer = tf.summary.FileWriter('log/', session.graph)

		cnt = 0

		while True:
			try:
				# Runs `features` TF op and return a tuple with Numpy arrays
				features_input, features_target = session.run(features)
				#vysledek = session.run(moje_super_ops, feed_dict={m_nn_input: features_input, m_nn_target: features_target})

				cnt += 1

				# Outputs features_data[0] (sample_input) with shape (BATCH_SIZE,7)
				# and features_data[1] (sample_target) with shape (BATCH_SIZE,1)

				# print('Sample input ', features_data[0], ', sample target ', features_data[1])
			except OutOfRangeError:
				# Ends the cycle if we run out of data samples
				break

		print("Count: ", cnt)

		writer.close()