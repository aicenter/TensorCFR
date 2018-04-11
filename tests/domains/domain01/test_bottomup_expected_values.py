import numpy as np
import tensorflow as tf

from src.constants import SMALL_ERROR_TOLERANCE, LARGE_ERROR_TOLERANCE
from src.domains.domain01.bottomup_expected_values import get_expected_values


class TestBottomUpExpectedValues(tf.test.TestCase):
	def setUp(self):
		self.expected_values = get_expected_values()

	def test_level_0(self):
		expected_output = np.array(83.9625)

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.expected_values[0])

			self.assertNDArrayNear(output, expected_output, LARGE_ERROR_TOLERANCE)

	def test_level_1(self):
		expected_output = np.array([25.45, 95.5, 154.2, 214.2, 210.45])

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.expected_values[1])

			self.assertNDArrayNear(output, expected_output, LARGE_ERROR_TOLERANCE)

	def test_level_2(self):
		expected_output = np.array([[18.5, 33, 30],
									[77.5, 97.5, 0],
									[135, 159, 0],
									[195, 219, 0],
									[130, 275.5, 296]])

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.expected_values[2])

			self.assertNDArrayNear(output, expected_output, SMALL_ERROR_TOLERANCE)

	def test_level_3(self):
		expected_output = np.array([[[10, 20],
									 [30, 40],
									 [0, 0]],

									[[70, 80],
									 [90, 100],
									 [0, 0]],

									[[130, 140],
									 [150, 160],
									 [0, 0]],

									[[190, 200],
									 [210, 220],
									 [0, 0]],

									[[0, 0],
									 [270, 280],
									 [290, 300]]])

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.expected_values[3])

			self.assertNDArrayNear(output, expected_output, SMALL_ERROR_TOLERANCE)
