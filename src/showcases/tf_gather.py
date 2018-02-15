#!/usr/bin/env python3

import tensorflow as tf

from utils.tensor_utils import print_tensors

count_of_actions = 2
shape2x2 = [count_of_actions, count_of_actions]
tensor2x2 = tf.reshape(tf.range(0, 4), shape2x2, name="tensor2x2")
mask2x2 = tf.Variable([[0, 1],
                       [1, 0]],
                      name="mask2x2")
gathered_tensor = tf.gather(params=tensor2x2, indices=mask2x2, name="gathered_tensor")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [tensor2x2, mask2x2, gathered_tensor])
