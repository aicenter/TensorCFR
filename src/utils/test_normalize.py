import numpy as np
import tensorflow as tf

from utils.tensor_utils import normalize


class TestTensorNormalization(tf.test.TestCase):
	def setUp(self):
		self.error_tolerance = 0.0000001

		self.input_tensor_1dim = tf.constant(np.arange(4), tf.float32)
		self.input_tensor_2dim = tf.constant(np.arange(4).reshape(2, 2), tf.float64)
		self.input_tensor_3dim = tf.constant(np.arange(8).reshape(2, 2, 2), tf.float64)

	def test_vector_l1_norm(self):
		expected_output = np.array([0., 0.16666667, 0.33333333, 0.5])

		with self.test_session() as sess:
			output = sess.run(normalize(self.input_tensor_1dim))

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_matrix_l1_norm(self):
		expected_output = np.array([[0., 1.],
		                            [0.4, 0.6]])

		with self.test_session() as sess:
			output = sess.run(normalize(self.input_tensor_2dim))

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_3d_tensor_l1_norm(self):
		expected_output = np.array([[[0., 1.],
		                             [0.4, 0.6]],

		                            [[0.44444444, 0.55555556],
		                             [0.46153846, 0.53846154]]])

		with self.test_session() as sess:
			output = sess.run(normalize(self.input_tensor_3dim))

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_vector_l2_norm(self):
		expected_output = np.array([0., 0.26726124, 0.53452248, 0.80178373])

		with self.test_session() as sess:
			output = sess.run(normalize(self.input_tensor_1dim, order=2))

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_matrix_l2_norm(self):
		expected_output = np.array([[0., 1.],
		                            [0.5547002, 0.83205029]])

		with self.test_session() as sess:
			output = sess.run(normalize(self.input_tensor_2dim, order=2))

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_3d_tensor_l2_norm(self):
		expected_output = np.array([[[0., 1.],
		                             [0.5547002, 0.83205029]],

		                            [[0.62469505, 0.78086881],
		                             [0.65079137, 0.7592566]]])

		with self.test_session() as sess:
			output = sess.run(normalize(self.input_tensor_3dim, order=2))

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)
