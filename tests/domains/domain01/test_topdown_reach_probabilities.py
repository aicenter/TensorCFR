import numpy as np
import tensorflow as tf

from src.domains.domain01.topdown_reach_probabilities import get_reach_probabilities


class TestTopDownReachProbabilities(tf.test.TestCase):
	def setUp(self):
		self.error_tolerance = 0.0000001

		self.reach_probabilities = get_reach_probabilities()

	def test_level_0(self):
		expected_output = np.array([1])

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.reach_probabilities[0])

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_level_1(self):
		expected_output = np.array([0.5, 0.25, 0.1, 0.1, 0.05])

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.reach_probabilities[1])

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_level_2(self):
		expected_output = np.array([
									[0.5, 0.5, 0.5],
									[0.025, 0.225, 0],
									[0.02, 0.08, 0],
									[0.02, 0.08, 0],
									[0.015, 0.015, 0.015]
		])

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.reach_probabilities[2])

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_level_3(self):
		expected_output = np.array([[[0.5, 0.5],
									[0.35, 0.15],
									[0, 0]],

									[[0.025, 0.025],
									[0.225, 0.225],
									[0, 0]],

									[[0.01, 0.01],
									[0.008, 0.072],
									[0, 0]],

									[[0.01, 0.01],
									[0.008, 0.072],
									[0, 0]],

									[[0, 0],
									[0.015, 0.015],
									[0.006, 0.009]]])

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			output = sess.run(self.reach_probabilities[3])

			print(output)

			self.assertNDArrayNear(output, expected_output, self.error_tolerance)
