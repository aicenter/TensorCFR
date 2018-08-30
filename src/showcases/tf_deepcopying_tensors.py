#!/usr/bin/env python3

# from https://stackoverflow.com/questions/33717772/how-can-i-copy-a-variable-in-tensorflow
import tensorflow as tf

from src.utils.tf_utils import print_tensor

var = tf.Variable(0.9)
var2 = tf.Variable(0.0)
deepcopy_first_variable = var2.assign(var)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensor(sess, var2)
	print_tensor(sess, var2)
	sess.run(deepcopy_first_variable)
	print_tensor(sess, var2)
	print_tensor(sess, var2)
