from unittest import TestCase
from utils.tensor_utils import expanded_multiply, print_tensors
import tensorflow as tf


class TestExpandedMultiply(TestCase):
	def test_expanded_multiply_0D_by_1D(self):
		expandable_tensor = tf.Variable(2.0, name="expandable_tensor")
		expanded_tensor = tf.Variable([0.5, 0.25, 0.1, 0.1, 0.05], name="expandable_tensor")
		result = expanded_multiply(expandable_tensor=expandable_tensor, expanded_tensor=expanded_tensor,
		                           name="test_expanded_multiply_0D_by_1D")
		expected_result = tf.Variable([1.0, 0.5, 0.2, 0.2, 0.1], name="expected_result")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print_tensors(sess, [result, expected_result])
			tf.assert_equal(result, expected_result)

	def test_expanded_multiply_1D_by_2D(self):
		expandable_tensor = tf.Variable([0.5, 0.25, 0.1, 0.1, 0.05], name="expandable_tensor")
		expanded_tensor = tf.Variable([[1.0, 1.0, 1.0],
		                               [0.1, 0.9, 0.0],
		                               [0.2, 0.8, 0.0],
		                               [0.2, 0.8, 0.0],
		                               [0.3, 0.3, 0.3]],
		                              name="expandable_tensor")
		result = expanded_multiply(expandable_tensor=expandable_tensor, expanded_tensor=expanded_tensor,
		                           name="test_expanded_multiply_1D_by_2D")
		expected_result = tf.Variable([[0.5,     0.5, 0.5],
		                               [0.025, 0.225, 0.0],
		                               [0.02,   0.08, 0.0],
		                               [0.02,   0.08, 0.0],
		                               [0.015, 0.015, 0.015]], name="expected_result")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print_tensors(sess, [result, expected_result])
			# difference = tf.subtract(result, expected_result, name="difference")
			# print_tensors(sess, [difference])
			tf.assert_equal(result, expected_result)
