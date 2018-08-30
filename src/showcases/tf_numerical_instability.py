#!/usr/bin/env python3

import tensorflow as tf

from src.utils.tf_utils import print_tensors

a = tf.constant(0.8, dtype=tf.float64)
b = tf.constant(0.1, dtype=tf.float64)
result = tf.multiply(a, b, name="test_tensor")
expected_result = tf.constant(0.08, dtype=tf.float64, name="expected_result")
equality_of_results = tf.equal(result, expected_result, name="equality_of_results")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [a, b, result, expected_result, equality_of_results])
