#!/usr/bin/env python3

import tensorflow as tf

from src.utils.tf_utils import print_tensors

t = tf.constant(
		[
			[1, 2, 3],
			[4, 5, 6]
		]
)
paddings = tf.constant(
		[
			[1, 1],
			[2, 2]
		]
)
p = tf.pad(t, paddings, name="2D-padding")


t2 = tf.constant([10.6])
paddings2 = tf.constant(
		[
			[0, 6]
		]
)
p2 = tf.pad(t2, paddings2, name="1D-padding")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [p, p2])
