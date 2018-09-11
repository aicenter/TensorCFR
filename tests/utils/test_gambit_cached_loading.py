import os
import tensorflow as tf

from src.commons.constants import PROJECT_ROOT
from src.domains.FlattenedDomain import FlattenedDomain


class TestGambitCachedLoading(tf.test.TestCase):
	def setUp(self):
		path_to_domain_filename = os.path.join(
			PROJECT_ROOT,
			'doc',
			'goofspiel',
			'II-GS2.efg'
		)

		self.domain_from_hkl_load1 = FlattenedDomain.init_from_hkl_file(path_to_domain_filename)
		self.domain_from_hkl_load2 = FlattenedDomain.init_from_hkl_file(path_to_domain_filename)
		self.domain_from_gambit = FlattenedDomain.init_from_hkl_file(path_to_domain_filename)

		self.levels = self.domain_from_hkl_load1.levels

	def test_1(self):
		with self.test_session() as sess:
			self.assertAllEqual(self.domain_from_hkl_load1.utilities, self.domain_from_gambit.utilities)
			self.assertAllEqual(self.domain_from_hkl_load1.node_to_infoset, self.domain_from_gambit.node_to_infoset)

			for level in range(self.levels):
				self.assertAllEqual(sess.run(self.domain_from_hkl_load1.action_counts_of_inner_nodes[level]),
									sess.run(self.domain_from_gambit.action_counts_of_inner_nodes[level]))

				self.assertAllEqual(sess.run(self.domain_from_hkl_load1.mask_of_inner_nodes[level]),
									sess.run(self.domain_from_gambit.mask_of_inner_nodes[level]))

if __name__ == '__main__':
	tf.test.main()
