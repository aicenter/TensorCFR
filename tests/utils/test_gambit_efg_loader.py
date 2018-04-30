import os
import unittest

import numpy as np

from src.utils.gambit_efg_loader import GambitEFGLoader


class TestGambitEFGLoaderParse(unittest.TestCase):
	def setUp(self):
		# TODO implement
		pass


class TestGambitEFGLoaderDomain01(unittest.TestCase):
	def setUp(self):
		domain01_efg = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc',
		                            'domain01_via_gambit.efg')
		self.number_of_levels = 3
		self.domain = GambitEFGLoader(domain01_efg)

	def test_actions_per_level(self):
		expected_output = np.array([5, 3, 2])
		np.testing.assert_array_equal(self.domain.actions_per_levels, expected_output)

	def test_current_infoset_strategies(self):
		expected_output = [None] * 3
		expected_output[0] = np.array([[0.5, 0.25, 0.1, 0.1, 0.05]])
		expected_output[1] = np.array([[0.08333333333333333, 0.08333333333333333, 0.08333333333333333],
		                               [0.08333333333333333, 0.08333333333333333, 0.08333333333333333],
		                               [0.08333333333333333, 0.08333333333333333, 0.08333333333333333],
		                               [0.3333333333333333, 0.3333333333333333, 0.3333333333333333]])
		expected_output[2] = np.array([[0.0625, 0.0625],
		                               [0.0625, 0.0625],
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
		                                [30., 40.],
		                                [0., 0.]],

		                               [[70., 80.],
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
		expected_output[1] = np.zeros((4, 3))
		expected_output[2] = np.zeros((9, 2))

		for lvl in range(self.number_of_levels):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.positive_cumulative_regrets[lvl])

	def test_cumulative_regrets(self):
		expected_output = [None] * 4
		expected_output[0] = 0
		expected_output[1] = np.zeros((4, 3))
		expected_output[2] = np.zeros((9, 2))

		for lvl in range(self.number_of_levels):
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
		expected_output[2] = np.array([[0, 1, 5],
		                               [2, 2, 8],
		                               [3, 4, 8],
		                               [3, 4, 8],
		                               [5, 6, 7]])

		for lvl in range(self.number_of_levels):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.node_to_infoset[lvl])

	def test_infoset_acting_players(self):
		expected_output = [None] * 3
		expected_output[0] = 0
		expected_output[1] = np.array([1, 2, 2, 0])
		expected_output[2] = np.array([1, 2, 1, 2, 0, -1, 1, 2, -1])

		for lvl in range(self.number_of_levels):
			np.testing.assert_array_equal(expected_output[lvl], self.domain.infoset_acting_players[lvl])


if __name__ == '__main__':
	unittest.main()
