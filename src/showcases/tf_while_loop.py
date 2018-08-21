#!/usr/bin/env python3

import tensorflow as tf

i = tf.constant(1)
c = lambda i: tf.less(i, 1000)
b = lambda i: tf.add(i, 1)
r = tf.while_loop(c, b, [i])

with tf.Session() as sess:
	print(sess.run(i))
	print(sess.run(r))
	print(sess.run(i))
