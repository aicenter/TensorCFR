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
		domain_1 = self.domain_from_hkl_load1
		domain_2 = self.domain_from_hkl_load2

		with self.test_session() as sess:
			# list
			self.assertAllEqual(domain_1.action_counts, domain_2.action_counts)

			# list of tf.Variables
			self.assertAllEqual(domain_1.utilities, domain_2.utilities)
			self.assertAllEqual(domain_1.node_to_infoset, domain_2.node_to_infoset)
			self.assertAllEqual(domain_1.cumulative_infoset_strategies, domain_2.cumulative_infoset_strategies)

			# list of tf.Tensors
			for level in range(self.levels):
				self.assertAllEqual(sess.run(domain_1.action_counts_of_inner_nodes[level]),
									sess.run(domain_2.action_counts_of_inner_nodes[level]))

				self.assertAllEqual(sess.run(domain_1.mask_of_inner_nodes[level]),
									sess.run(domain_2.mask_of_inner_nodes[level]))

				self.assertAllEqual(sess.run(domain_1.action_counts_of_inner_nodes[level]),
									sess.run(domain_2.action_counts_of_inner_nodes[level]))

if __name__ == '__main__':
	tf.test.main()
