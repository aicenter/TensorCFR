import unittest
import os
import numpy as np

from src.utils.gambit_loader import GambitLoader


class TestGambitLoaderDomain01(unittest.TestCase):
	def setUp(self):
		domain01 = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'doc','domain01_via_gambit.efg')
		self.domain = GambitLoader(domain01)

	def test_node_to_infoset_level_0(self):
		expected_output = [0]
		np.testing.assert_array_equal(self.domain.node_to_infoset[0], expected_output)

	def test_node_to_infoset_level_1(self):
		expected_output = [3, 2, 1, 1, 0]
		np.testing.assert_array_equal(self.domain.node_to_infoset[1], expected_output)

	def test_node_to_infoset_level_2(self):
		expected_output = [6, 5, 7, 4, 4, 3, 2, 3, 2, 7, 1, 0]
		np.testing.assert_array_equal(self.domain.node_to_infoset[2], expected_output)