#!/usr/bin/env python3

import numpy as np
# taken from https://stackoverflow.com/questions/43263017/variables-with-dynamic-shape-tensorflow
import tensorflow as tf

v = tf.Variable([], validate_shape=False)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(v, feed_dict={v: np.zeros((3, 4))}))
	print(sess.run(v, feed_dict={v: np.ones((2, 2))}))
