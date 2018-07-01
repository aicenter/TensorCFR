import numpy as np
import tensorflow as tf

from src.utils.cfr_utils import get_parents_from_action_counts


class TestCFRUtils(tf.test.TestCase):
	def test_get_parents_from_action_counts(self):
		"""
		Test on `domains.hunger_games`
		"""
		self.action_counts = [
			[2],
			[1, 6],
			[4, 0, 0, 0, 0, 0, 0],
			[3, 3, 2, 2],
			[2] * 10,
			[0] * 20
		]
		self.expected_parents = [
			[np.nan],
			[0, 0],
			[0, 1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
			[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
		]
		self.parents = get_parents_from_action_counts(self.action_counts)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(self.parents), len(self.expected_parents))
			for i in range(len(self.parents)):
				tf.assert_equal(self.parents[i], self.expected_parents[i])
