import numpy as np
import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.commons.constants import SMALL_ERROR_TOLERANCE, LARGE_ERROR_TOLERANCE
from src.domains.domain01.Domain01 import get_domain01


class TestBottomUpExpectedValues(tf.test.TestCase):
	def setUp(self):
		self.domain01 = get_domain01()

	def test_level_0(self):
		expected_output = np.array(83.9625)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFR(self.domain01)
			expected_values = sess.run(tensorcfr.get_expected_values())
			self.assertNDArrayNear(expected_values[0], expected_output, LARGE_ERROR_TOLERANCE)

	def test_level_1(self):
		expected_output = np.array([25.45, 95.5, 154.2, 214.2, 210.45])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFR(self.domain01)
			expected_values = sess.run(tensorcfr.get_expected_values())
			self.assertNDArrayNear(expected_values[1], expected_output, LARGE_ERROR_TOLERANCE)

	def test_level_2(self):
		expected_output = np.array([[18.5, 33, 30],
									[77.5, 97.5, 0],
									[135, 159, 0],
									[195, 219, 0],
									[130, 275.5, 296]])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFR(self.domain01)
			expected_values = sess.run(tensorcfr.get_expected_values())
			self.assertNDArrayNear(expected_values[2], expected_output, SMALL_ERROR_TOLERANCE)

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
			tensorcfr = TensorCFR(self.domain01)
			expected_values = sess.run(tensorcfr.get_expected_values())
			self.assertNDArrayNear(expected_values[3], expected_output, SMALL_ERROR_TOLERANCE)


if __name__ == "__main__":
	tf.test.main()
