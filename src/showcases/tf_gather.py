#!/usr/bin/env python3

import tensorflow as tf

from src.utils.tf_utils import print_tensors

shape2x2 = [2, 2]
tensor2x2 = tf.reshape(tf.range(0, 4), shape2x2, name="tensor2x2")
mask2x2 = tf.Variable([[0, 1],
                       [1, 0]],
                      name="mask2x2")
gathered_tensor = tf.gather(params=tensor2x2, indices=mask2x2, name="gathered_tensor")
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [tensor2x2, mask2x2, gathered_tensor])


shape3x3 = [3, 3]
tensor3x3 = tf.reshape(tf.range(0, 9), shape3x3, name="tensor3x3")
mask2x2 = tf.Variable([[2, 1],
                       [1, 0]],
                      name="mask2x2")
gathered_tensor = tf.gather(params=tensor3x3, indices=mask2x2, name="gathered_tensor")
with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [tensor3x3, mask2x2, gathered_tensor])

# negative indices don't work in tf.gather
# shape3x2 = [3, 2]
# tensor3x2 = tf.reshape(tf.range(0, 6), shape3x2, name="tensor3x2")
# mask2x2_negative = tf.Variable([[0, -1],
#                                 [-2, -3]],
#                                name="mask2x2")
# gathered_tensor = tf.gather(params=tensor3x2, indices=mask2x2_negative, name="gathered_tensor")
# with tf.Session() as sess:
# 	sess.run(tf.global_variables_initializer())
# 	print_tensors(sess, [tensor3x2, mask2x2_negative, gathered_tensor])
