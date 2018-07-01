import numpy as np
import tensorflow as tf

from src.utils.cfr_utils import get_parents_from_action_counts, get_node_types_from_action_counts


class TestCFRUtils(tf.test.TestCase):
	def test_get_parents_from_action_counts(self):
		"""
		Test on `domains.hunger_games`
		"""
		action_counts = [
			[2],
			[1, 6],
			[4, 0, 0, 0, 0, 0, 0],
			[3, 3, 2, 2],
			[2] * 10,
			[0] * 20
		]
		expected_node_types = [
			[np.nan],
			[0, 0],
			[0, 1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
			[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
		]
		parents = get_parents_from_action_counts(action_counts)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(parents), len(expected_node_types))
			for i in range(len(parents)):
				tf.assert_equal(parents[i], expected_node_types[i])

	def test_get_node_types_from_action_counts(self):
		"""
		Test on `domains.hunger_games`
		"""
		action_counts = [
			[2],
			[1, 6],
			[4, 0, 0, 0, 0, 0, 0],
			[3, 3, 2, 2],
			[2] * 10,
			[0] * 20
		]
		expected_node_types = [
			[0],
			[0, 0],
			[0, 1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		]
		node_types = get_node_types_from_action_counts(action_counts)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(node_types), len(expected_node_types))
			for i in range(len(node_types)):
				tf.assert_equal(node_types[i], expected_node_types[i])
