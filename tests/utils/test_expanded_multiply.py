import tensorflow as tf

from src.constants import SMALL_ERROR_TOLERANCE
from src.utils.tensor_utils import expanded_multiply, print_tensors


class TestExpandedMultiply(tf.test.TestCase):
	def setUp(self):
		self.tolerance = SMALL_ERROR_TOLERANCE

	def test_expanded_multiply_0D_by_1D(self):
		expandable_tensor = tf.Variable(2.0, name="expandable_tensor")
		expanded_tensor = tf.Variable([0.5, 0.25, 0.1, 0.1, 0.05], name="expandable_tensor")
		result = expanded_multiply(expandable_tensor=expandable_tensor, expanded_tensor=expanded_tensor,
		                           name="test_expanded_multiply_0D_by_1D")
		expected_result = tf.Variable([1.0, 0.5, 0.2, 0.2, 0.1], name="expected_result")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print_tensors(sess, [result, expected_result])
			self.assertAllEqual(result.eval(), expected_result.eval())

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
			difference = tf.subtract(result, expected_result, name="difference")
			print_tensors(sess, [difference])
			self.assertNDArrayNear(result.eval(), expected_result.eval(), err=self.tolerance)


if __name__ == '__main__':
	tf.test.main()
