#!/usr/bin/env python3

import tensorflow as tf

count_of_actions = 3
shape3x3 = [count_of_actions, count_of_actions, count_of_actions]
tensor3x3 = tf.reshape(tf.range(0, 27), shape3x3)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(tensor3x3))
	print(sess.run(tensor3x3[2][0]))
	print(sess.run(tensor3x3[2][1]))
	print(sess.run(tensor3x3[2][2]))
