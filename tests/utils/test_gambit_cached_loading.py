import os
import unittest
import numpy as np

from src.commons.constants import PROJECT_ROOT
from src.utils.gambit_flattened_domains.loader import GambitLoader, GambitLoaderCached


class TestGambitCachedLoading(unittest.TestCase):
	def setUp(self):
		path_to_domain_filename = os.path.join(
			PROJECT_ROOT,
			'doc',
			'goofspiel',
			'II-GS2.efg'
		)

		self.domain_from_hkl_load1 = GambitLoaderCached(path_to_domain_filename)
		self.domain_from_hkl_load2 = GambitLoaderCached(path_to_domain_filename)
		self.domain_from_gambit = GambitLoader(path_to_domain_filename)

		self.levels = self.domain_from_gambit.number_of_levels
		print(self.levels)

	def test_1(self):
		domain1 = self.domain_from_hkl_load1
		domain2 = self.domain_from_hkl_load2

		# list
		np.testing.assert_array_equal(domain1.domain_parameters, domain2.domain_parameters)
		# list of lists
		np.testing.assert_array_equal(domain1.node_to_infoset, domain2.node_to_infoset)
		np.testing.assert_array_equal(domain1.number_of_nodes_actions, domain2.number_of_nodes_actions)
		np.testing.assert_array_equal(domain1.utilities, domain2.utilities)



if __name__ == '__main__':
	unittest.main()
