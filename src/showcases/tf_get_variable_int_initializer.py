#!/usr/bin/env python3

import tensorflow as tf


if __name__ == '__main__':
	with tf.variable_scope("domain_definitions", reuse=tf.AUTO_REUSE):
		current_opponent = tf.get_variable(
				"current_opponent",
				initializer=2,
				dtype=tf.int32,   # <- initializer defaults to floats!
		)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		with tf.variable_scope("domain_definitions", reuse=True):
			sess.run(
					tf.get_variable(
							"current_opponent",
							dtype=tf.int32,   # Without this line, you get a ValueError
					)
			)
