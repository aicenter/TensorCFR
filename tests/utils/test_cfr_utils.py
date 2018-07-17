import numpy as np
import tensorflow as tf

from src.utils.cfr_utils import get_parents_from_action_counts, get_node_types_from_action_counts, \
	distribute_strategies_to_nodes
from src.utils.tensor_utils import print_tensors


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

	def test_distribute_strategies_to_nodes(self):
		tf.test.TestCase.skipTest(
			self,
			reason=
			"""
The test `test_distribute_strategies_to_nodes` does not work on the CPU version of TensorFlow. Try with:

```python
with self.test_session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
```
			"""
		)

		"""
		Test on `domains.hunger_games`
		"""
		# taken from hunger_games.initial_infoset_strategies, see `doc/hunger_games/hunger_games_via_gambit.png`
		infoset_strategies = [
			tf.Variable([[0.1, 0.9]],
			            name="infoset_uniform_strategies_lvl0"),
			tf.Variable([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			             [0.1, 0.1, 0.1, 0.0, 0.2, 0.5]],
			            name="infoset_uniform_strategies_lvl1"),
			tf.Variable([[0.1, 0.2, 0.0, 0.7]],
			            name="infoset_uniform_strategies_lvl2"),
			tf.Variable([[0.1, 0.0, 0.9],
			             [0.2, 0.8, 0.0]],
			            name="infoset_uniform_strategies_lvl3"),
			tf.Variable([[0.1, 0.9],
			             [0.2, 0.8],
			             [0.3, 0.7],
			             [0.4, 0.6],
			             [0.5, 0.5],
			             [0.6, 0.4],
			             [0.7, 0.3],
			             [0.8, 0.2],
			             [0.9, 0.1],
			             [1.0, 0.0]],
			            name="infoset_uniform_strategies_lvl4")
		]
		node_to_infoset = [
			tf.Variable(0,
			            name="node_to_infoset_lvl0"),
			tf.Variable([0, 1],
			            name="node_to_infoset_lvl1"),
			tf.Variable([0, 1, 1, 1, 1, 1, 1],
			            name="node_to_infoset_lvl2"),
			tf.Variable([0, 0, 1, 1],
			            name="node_to_infoset_lvl3"),
			tf.Variable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
			            name="node_to_infoset_lvl4")
		]
		nodal_strategies = [
			distribute_strategies_to_nodes(
					infoset_strategies[level],
					node_to_infoset[level],
					"nodal_strategies_lvl{}".format(level)
			)
			for level in range(len(infoset_strategies))
		]
		expected_nodal_strategies = [
			[[0.1, 0.9]],   # level 0
			[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],   # level 1
			 [0.1, 0.1, 0.1, 0.0, 0.2, 0.5]],
			[[0.1, 0.2, 0.,  0.7],   # level 2
			 [0.,  0.,  0.,  0., ],
			 [0.,  0.,  0.,  0., ],
			 [0.,  0.,  0.,  0., ],
			 [0.,  0.,  0.,  0., ],
			 [0.,  0.,  0.,  0., ],
			 [0.,  0.,  0.,  0., ]],
			[[0.1, 0.,  0.9],   # level 3
			 [0.1, 0.,  0.9],
			 [0.2, 0.8, 0.],
			 [0.2, 0.8, 0.]],
			[[0.1, 0.9],   # level 4
			 [0.2, 0.8],
			 [0.3, 0.7],
			 [0.4, 0.6],
			 [0.5, 0.5],
			 [0.6, 0.4],
			 [0.7, 0.3],
			 [0.8, 0.2],
			 [0.9, 0.1],
			 [1.,  0.]]
		]

		# TODO here
		# updating_player =
		# acting_players =

		# with self.test_session(config=tf.ConfigProto(device_count={'GPU': 0})) as sess:
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(nodal_strategies), len(expected_nodal_strategies))
			for i in range(len(nodal_strategies)):
				tf.assert_equal(nodal_strategies[i], expected_nodal_strategies[i])
			print_tensors(sess, nodal_strategies)
