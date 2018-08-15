import numpy as np
import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.algorithms.tensorcfr_fixed_trunk_strategies.TensorCFRFixedTrunkStrategies import TensorCFRFixedTrunkStrategies
from src.commons.constants import SMALL_ERROR_TOLERANCE, ALL_PLAYERS
from src.domains.domain01.Domain01 import get_domain01
from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit


class TestTopDownReachProbabilities(tf.test.TestCase):
	def setUp(self):
		self.error_tolerance = SMALL_ERROR_TOLERANCE
		self.domain01 = get_domain01()
		self.flattened_domain01 = get_flattened_domain01_from_gambit()

	def test_level_0_via_dense_tensorcfr(self):
		expected_output = np.array([1])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFR(self.domain01)
			reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities())
			self.assertNDArrayNear(reach_probabilities[0], expected_output, self.error_tolerance)

	def test_level_1_via_dense_tensorcfr(self):
		expected_output = np.array([0.5, 0.25, 0.1, 0.1, 0.05])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFR(self.domain01)
			reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities())
			self.assertNDArrayNear(reach_probabilities[1], expected_output, self.error_tolerance)

	def test_level_2_via_dense_tensorcfr(self):
		expected_output = np.array([
			[0.5, 0.5, 0.5],
			[0.025, 0.225, 0],
			[0.02, 0.08, 0],
			[0.02, 0.08, 0],
			[0.015, 0.015, 0.015]
		])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFR(self.domain01)
			reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities())
			self.assertNDArrayNear(reach_probabilities[2], expected_output, self.error_tolerance)

	def test_level_3_via_dense_tensorcfr(self):
		expected_output = np.array([
			[
				[0.5, 0.5],
				[0.35, 0.15],
				[0, 0]
			],
			[
				[0.025, 0.025],
				[0.225, 0.225],
				[0, 0]
			],
			[
				[0.01, 0.01],
				[0.008, 0.072],
				[0, 0]
			],
			[
				[0.01, 0.01],
				[0.008, 0.072],
				[0, 0]
			],
			[
				[0, 0],
				[0.015, 0.015],
				[0.006, 0.009]
			]
		])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFR(self.domain01)
			reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities())
			self.assertNDArrayNear(reach_probabilities[3], expected_output, self.error_tolerance)

	def test_level_0_via_tensorcfr_fixed_trunk_for_all_players(self):
		expected_output = np.array([1])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFRFixedTrunkStrategies(self.flattened_domain01)
			reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities(for_player=ALL_PLAYERS))
			self.assertNDArrayNear(reach_probabilities[0], expected_output, self.error_tolerance)

	def test_level_1_via_tensorcfr_fixed_trunk(self):
		expected_output = np.array([0.5, 0.25, 0.1, 0.1, 0.05])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFRFixedTrunkStrategies(self.flattened_domain01)
			reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities(for_player=ALL_PLAYERS))
			self.assertNDArrayNear(reach_probabilities[1], expected_output, self.error_tolerance)

	def test_level_2_via_tensorcfr_fixed_trunk(self):
		expected_output = np.array([
			0.16666667, 0.16666667, 0.16666667,
			0.125, 0.125,
			0.05, 0.05,
			0.05, 0.05,
			0.01666667, 0.01666667, 0.01666667
		])
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			tensorcfr = TensorCFRFixedTrunkStrategies(self.flattened_domain01)
			reach_probabilities = sess.run(tensorcfr.get_nodal_reach_probabilities(for_player=ALL_PLAYERS))
			self.assertNDArrayNear(reach_probabilities[2], expected_output, self.error_tolerance)


if __name__ == "__main__":
	tf.test.main()
