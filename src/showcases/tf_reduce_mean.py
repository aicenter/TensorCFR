#!/usr/bin/env python3

# from https://www.dotnetperls.com/reduce-mean-tensorflow
import tensorflow as tf

from src.utils.tf_utils import print_tensors

array_of_values = [[1, 2], [3, 4], [5, 6]]
values1 = tf.constant(
	array_of_values,
	name="values1"
)
values1_reduced = tf.reduce_mean(values1, name="values1_reduced", axis=0)
values2 = tf.placeholder(tf.float32, [None, 2], name="values2")
values2_reduced = tf.reduce_mean(values2, name="values2_reduced", axis=0)

with tf.Session() as sess:
	np_values2, np_values2_reduced = sess.run(
		[values2, values2_reduced],
		feed_dict={values2: array_of_values}
	)
	print_tensors(sess, [values1, values1_reduced])
	print(np_values2)
	print(np_values2_reduced)
