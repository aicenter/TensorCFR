from unittest import TestCase
from utils.tensor_utils import masked_assign, print_tensors
import tensorflow as tf


class TestMaskedAssign(TestCase):
	def test_masked_assign(self):
		shape2x2x2 = [2, 2, 2]
		tensor = tf.Variable(tf.reshape(tf.range(1.0, 9.0), shape2x2x2), name="tensor")
		mask = tf.less(tensor, 6.0, name="mask")
		new_values = tf.fill(shape2x2x2, -0.5)
		masked_assignment = masked_assign(ref=tensor, mask=mask, value=new_values)
		expected_result = tf.constant( [[[-0.5, -0.5],
		                                 [-0.5, -0.5]],
		                                [[-0.5,    6],
		                                 [    7,   8]]], dtype=tensor.dtype, name="expected_result")
		are_tensors_equal = tf.reduce_all(tf.equal(x=tensor, y=expected_result))
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertFalse(sess.run(are_tensors_equal))
			sess.run(masked_assignment)
			print_tensors(sess, [tensor, mask, masked_assignment, expected_result, are_tensors_equal])
			self.assertTrue(sess.run(are_tensors_equal))
