#!/usr/bin/env python3

import tensorflow as tf

from utils.tensor_utils import print_tensors

result = tf.multiply(tf.constant(0.8), tf.constant(0.1), name="test_tensor")
expected_result = tf.constant(0.08, name="expected_result")
equality_of_results = tf.equal(result, expected_result, name="equality_of_results")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [result, expected_result, equality_of_results])
