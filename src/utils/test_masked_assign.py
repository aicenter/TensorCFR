from unittest import TestCase
from utils.tensor_utils import masked_assign, print_tensors
import tensorflow as tf


class TestMaskedAssign(TestCase):
	def setUp(self):
		self.shape2x2x2 = [2, 2, 2]
		self.tensor = tf.Variable(tf.reshape(tf.range(1.0, 9.0), self.shape2x2x2), name="tensor")
		self.mask = tf.less(self.tensor, 6.0, name="mask")
		self.new_values = tf.fill(self.shape2x2x2, -0.5)
		self.masked_assignment = masked_assign(ref=self.tensor, mask=self.mask, value=self.new_values)

	def test_range_2x2x2(self):
		expected_result = tf.constant([[[-0.5, -0.5],
		                                 [-0.5, -0.5]],
		                                [[-0.5,  6.0],
		                                 [ 7.0,  8.0]]], dtype=self.tensor.dtype, name="expected_result")
		are_tensors_equal = tf.reduce_all(tf.equal(x=self.tensor, y=expected_result))
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertFalse(sess.run(are_tensors_equal))
			sess.run(self.masked_assignment)
			print_tensors(sess, [self.tensor, self.mask, self.masked_assignment, expected_result, are_tensors_equal])
			self.assertTrue(sess.run(are_tensors_equal))
