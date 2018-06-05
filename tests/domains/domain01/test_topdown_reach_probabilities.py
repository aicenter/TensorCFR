import numpy as np
import tensorflow as tf

from src.commons.constants import SMALL_ERROR_TOLERANCE
from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import get_domain01
# from src.domains.domain01.topdown_reach_probabilities import get_nodal_reach_probabilities


class TestTopDownReachProbabilities(tf.test.TestCase):
	def setUp(self):
		self.error_tolerance = SMALL_ERROR_TOLERANCE
		# self.reach_probabilities = get_nodal_reach_probabilities()
		domain01 = get_domain01()

		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())

			tensorcfr = TensorCFR(domain01)

			self.reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities())
			print(self.reach_probabilities)


	# def test_level_0(self):
	# 	expected_output = np.array([1])
	#
	# 	with self.test_session() as sess:
	# 		sess.run(tf.global_variables_initializer())
	#
	# 		output = sess.run(self.reach_probabilities[0])
	#
	# 		self.assertNDArrayNear(output, expected_output, self.error_tolerance)

	def test_level_1(self):
		expected_output = np.array([0.5, 0.25, 0.1, 0.1, 0.05])



		# with self.test_session() as sess:
		# 	sess.run(tf.global_variables_initializer())
		#
		# 	tensorcfr = TensorCFR(self.domain01)
		# 	self.reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities())
		#
		# 	print(type(expected_output))
		# 	print(type(self.reach_probabilities[1]))
		self.assertNDArrayNear(self.reach_probabilities[1], expected_output, self.error_tolerance)

	# def test_level_2(self):
	# 	expected_output = np.array([[0.5, 0.5, 0.5],
	# 								[0.025, 0.225, 0],
	# 								[0.02, 0.08, 0],
	# 								[0.02, 0.08, 0],
	# 								[0.015, 0.015, 0.015]])
	#
	# 	with self.test_session() as sess:
	# 		sess.run(tf.global_variables_initializer())
	#
	# 		output = sess.run(self.reach_probabilities[2])
	#
	# 		self.assertNDArrayNear(output, expected_output, self.error_tolerance)
	#
	# def test_level_3(self):
	# 	expected_output = np.array([[[0.5, 0.5],
	# 								 [0.35, 0.15],
	# 								 [0, 0]],
	#
	# 								[[0.025, 0.025],
	# 								 [0.225, 0.225],
	# 								 [0, 0]],
	#
	# 								[[0.01, 0.01],
	# 								 [0.008, 0.072],
	# 								 [0, 0]],
	#
	# 								[[0.01, 0.01],
	# 								 [0.008, 0.072],
	# 								 [0, 0]],
	#
	# 								[[0, 0],
	# 								 [0.015, 0.015],
	# 								 [0.006, 0.009]]])
	#
	# 	with self.test_session() as sess:
	# 		sess.run(tf.global_variables_initializer())
	#
	# 		output = sess.run(self.reach_probabilities[3])
	#
	# 		self.assertNDArrayNear(output, expected_output, self.error_tolerance)

if __name__ == "__main__":
	tf.test.main()
