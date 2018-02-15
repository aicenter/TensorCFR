#!/usr/bin/env python3

import tensorflow as tf

from utils.tensor_utils import print_tensors

shape = [3, 3, 3]
tensor = tf.Variable(tf.reshape(tf.range(0.0, 27), shape), name="tensor")
new_values = tf.fill(tensor.get_shape(), -0.5)
assign_op = tf.assign(ref=tensor, value=new_values, name="assign_op")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [tensor, assign_op, tensor])
