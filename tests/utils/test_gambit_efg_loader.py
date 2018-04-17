import os
import numpy as np
import unittest
import tensorflow as tf

from src.utils.gambit_efg_loader import GambitEFGLoader


class TestGambbitEFGLoaderParse(unittest.TestCase):
	def setUp(self):
		pass

	"""
	def test_parse_chance_node(self):
		input_str = 'c "" 1 "" { "Ea (0.05)" 0.05 "Da (0.1)" 0.1 "Ca (0.1)" 0.1 "Ba (0.25)" 0.25 "Aa (0.5)" 0.5 } 1 "" { 0, 0 }'

		gambit_loader = GambitEFGLoader("dummy")

		expected_output = {
			'payoffs': [0, 0],
			'outcome_name': '',
			'outcome': '1',
			'type': 'c',
			'name': '',
			'information_set_number': '1',
			'information_set_name': '',
			'actions': [
				{'probability': 0.05, 'name': 'Ea (0.05)'},
				{'probability': 0.1, 'name': 'Da (0.1)'},
				{'probability': 0.1, 'name': 'Ca (0.1)'},
				{'probability': 0.25, 'name': 'Ba (0.25)'},
				{'probability': 0.5, 'name': 'Aa (0.5)'}
			]
		}

		self.assertEqual(gambit_loader.parse_chance_node(input_str), expected_output)
	"""

class TestGambitEFGLoaderDomain01(unittest.TestCase):
	def setUp(self):
		domain01_efg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc', 'domain01_via_gambit.efg')
		self.number_of_levels = 3
		self.domain = GambitEFGLoader('/home/ruda/Documents/Projects/tensorcfr/TensorCFR/src/utils/domain01_via_gambit.efg')

	def test_actions_per_level(self):
		expected_output = np.array([5, 3, 2])
		np.testing.assert_array_equal(self.domain.max_actions_per_level, expected_output)

	def test_current_infoset_strategies(self):
		expected_output = [None] * 3
		expected_output[0] = np.array([[0.5, 0.25, 0.1, 0.1, 0.05]])
		expected_output[1] = np.array([[0.08333333333333333, 0.08333333333333333, 0.08333333333333333],
									   [0.08333333333333333, 0.08333333333333333, 0.08333333333333333],
									   [0.08333333333333333, 0.08333333333333333, 0.08333333333333333],
									   [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]])
		expected_output[2] = np.array([[0.0625, 0.0625],
									   [0.0625, 0.0625],
									   [0., 0.],
									   [0.0625, 0.0625],
									   [0.0625, 0.0625],
									   [0.1, 0.9],
									   [0., 0.],
									   [0.0625, 0.0625],
									   [0.0625, 0.0625],
									   [0., 0.]])

		for lvl in range(self.number_of_levels):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.current_infoset_strategies[lvl])

	def test_utilities(self):
		expected_output = [None] * 4
		expected_output[0] = 0
		expected_output[1] = np.array([0, 0, 0, 0, 0])
		expected_output[2] = np.array([[0, 0, 30],
									   [0, 0, 0],
									   [0, 0, 0],
									   [0, 0, 0],
									   [130, 0, 0]])
		expected_output[3] = np.array([[[10., 20.],
										[30.,40.],
										[0., 0.]],

									   [[ 70.,  80.],
										[90., 100.],
										[0., 0.]],

									   [[130., 140.],
										[150., 160.],
										[0., 0.]],

									   [[190., 200.],
										[210., 220.],
										[0., 0.]],

									   [[0., 0.],
										[270., 280.],
										[290., 300.]]])

		for lvl in range(self.number_of_levels + 1):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.utilities[lvl])

	def test_positive_cumulative_regrets(self):
		expected_output = [None] * 4
		expected_output[0] = 0
		expected_output[1] = np.zeros((5,))
		expected_output[2] = np.zeros((5, 3))
		expected_output[3] = np.zeros((5, 3, 2))

		for lvl in range(self.number_of_levels + 1):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.positive_cumulative_regrets[lvl])

	def test_cumulative_regrets(self):
		expected_output = [None] * 4
		expected_output[0] = 0
		expected_output[1] = np.zeros((5,))
		expected_output[2] = np.zeros((5, 3))
		expected_output[3] = np.zeros((5, 3, 2))

		for lvl in range(self.number_of_levels + 1):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.cumulative_regrets[lvl])

	def test_node_types(self):
		expected_output = [None] * 4
		expected_output[0] = 0
		expected_output[1] = np.array([0, 0, 0, 0, 0])
		expected_output[2] = np.array([[0, 0, 1],
									   [0, 0, 2],
									   [0, 0, 2],
									   [0, 0, 2],
									   [1, 0, 0]])
		expected_output[3] = np.array([[[1, 1],
										[1, 1],
										[2, 2]],

									   [[1, 1],
										[1, 1],
										[2, 2]],

									   [[1, 1],
										[1, 1],
										[2, 2]],

									   [[1, 1],
										[1, 1],
										[2, 2]],

									   [[2, 2],
										[1, 1],
										[1, 1]]])

		for lvl in range(self.number_of_levels + 1):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.node_types[lvl])

	def test_node_to_infoset(self):
		expected_output = [None] * 3
		expected_output[0] = 0
		expected_output[1] = np.array([0, 1, 2, 2, 3])
		expected_output[2] = np.array([[0., 1., 2.],
									   [3., 3., 9.],
									   [4., 5., 9.],
									   [4., 5., 9.],
									   [6., 7., 8.]])

		for lvl in range(self.number_of_levels):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.node_to_infoset[lvl])

	def test_infoset_acting_players(self):
		expected_output = [None] * 3
		expected_output[0] = 0
		expected_output[1] = np.array([1, 2, 2, 0])
		expected_output[2] = np.array([1, 2, -1, 1, 2, 0, -1, 1, 2, -1])

		for lvl in range(self.number_of_levels):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.infoset_acting_players[lvl])

if __name__ == '__main__':
	unittest.main()