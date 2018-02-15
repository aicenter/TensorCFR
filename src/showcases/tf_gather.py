#!/usr/bin/env python3

import tensorflow as tf

count_of_actions = 2
shape2x2 = [count_of_actions, count_of_actions]
tensor2x2 = tf.reshape(tf.range(0, 4), shape2x2)
mask2x2 = tf.Variable([[0, 1],
                       [1, 0]])
gathered_tensor = tf.gather(params=tensor2x2, indices=mask2x2)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# print(sess.run(tensor2x2))
	# print(sess.run(mask2x2))
	print(sess.run(gathered_tensor))
	# print(sess.run(tensor3x3[2][0]))
	# print(sess.run(tensor3x3[2][1]))
	# print(sess.run(tensor3x3[2][2]))
