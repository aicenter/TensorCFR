#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


def body(x):
	a = tf.random_uniform(shape=[2, 2], dtype=tf.int32, maxval=100)
	b = tf.constant(np.array([[1, 2], [3, 4]]), dtype=tf.int32)
	c = a + b
	return tf.nn.relu(x + c)


def condition(x):
	return tf.reduce_sum(x) < 100


if __name__ == '__main__':
	i = tf.constant(1)
	c = lambda i: tf.less(i, 1000)
	b = lambda i: tf.add(i, 1)
	r = tf.while_loop(c, b, [i])
	with tf.Session() as sess:
		print(sess.run(i))
		print(sess.run(r))
		print(sess.run(i))

	x = tf.Variable(tf.constant(0, shape=[2, 2]))
	with tf.Session():
		tf.initialize_all_variables().run()
		result = tf.while_loop(condition, body, [x])
		print(result.eval())
