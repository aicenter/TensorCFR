from pprint import pprint

import numpy as np
import tensorflow as tf

from src.commons.constants import INFOSET_FOR_TERMINAL_NODES
from src.utils.cfr_utils import get_parents_from_action_counts, get_node_types_from_action_counts, \
	distribute_strategies_to_nodes, distribute_strategies_to_inner_nodes
from src.utils.tensor_utils import print_tensors


class TestCFRUtils(tf.test.TestCase):
	def setUp(self):
		# taken from hunger_games.initial_infoset_strategies, see `doc/hunger_games/hunger_games_via_gambit.png`
		self.action_counts = [
			[2],
			[1, 6],
			[4, 0, 0, 0, 0, 0, 0],
			[3, 3, 2, 2],
			[2] * 10,
			[0] * 20
		]
		self.infoset_strategies = [
			tf.Variable([[0.1, 0.9]],
			            name="infoset_strategies_lvl0"),
			tf.Variable([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
			             [0.1, 0.1, 0.1, 0.0, 0.2, 0.5]],
			            name="infoset_strategies_lvl1"),
			tf.Variable([[0.1, 0.2, 0.0, 0.7]],
			            name="infoset_strategies_lvl2"),
			tf.Variable([[0.1, 0.0, 0.9],
			             [0.2, 0.8, 0.0]],
			            name="infoset_strategies_lvl3"),
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
			            name="infoset_strategies_lvl4")
		]
		self.node_to_infoset = [
			tf.Variable([0],
			            name="node_to_infoset_lvl0"),
			tf.Variable([0, 1],
			            name="node_to_infoset_lvl1"),
			tf.Variable([0, INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES,
			             INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES],
			            name="node_to_infoset_lvl2"),
			tf.Variable([0, 0, 1, 1],
			            name="node_to_infoset_lvl3"),
			tf.Variable([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
			            name="node_to_infoset_lvl4")
		]
		self.expected_nodal_strategies = [
			[[0.1, 0.9]],  # level 0
			[[1.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # level 1
			 [0.1, 0.1, 0.1, 0.0, 0.2, 0.5]],
			[[0.1, 0.2, 0., 0.7]],  # level 2
			[[0.1, 0., 0.9],  # level 3
			 [0.1, 0., 0.9],
			 [0.2, 0.8, 0.],
			 [0.2, 0.8, 0.]],
			[[0.1, 0.9],  # level 4
			 [0.2, 0.8],
			 [0.3, 0.7],
			 [0.4, 0.6],
			 [0.5, 0.5],
			 [0.6, 0.4],
			 [0.7, 0.3],
			 [0.8, 0.2],
			 [0.9, 0.1],
			 [1., 0.]]
		]
		self.mask_of_inner_nodes = [
			tf.greater(
				action_count,
				0,
				name="mask_of_inner_nodes_lvl{}".format(level)
			)
			for level, action_count in enumerate(self.action_counts)
		]

	def test_get_parents_from_action_counts(self):
		"""
		Test on `domains.hunger_games`
		"""
		expected_parents = [
			[np.nan],
			[0, 0],
			[0, 1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0],
			[0, 0, 0, 1, 1, 1, 2, 2, 3, 3],
			[0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9],
		]
		parents = get_parents_from_action_counts(self.action_counts)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(parents), len(expected_parents))
			for i in range(len(parents)):
				tf.assert_equal(parents[i], expected_parents[i])

	def test_get_node_types_from_action_counts(self):
		"""
		Test on `domains.hunger_games`
		"""
		expected_node_types = [
			[0],
			[0, 0],
			[0, 1, 1, 1, 1, 1, 1],
			[0, 0, 0, 0],
			[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
			[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
		]
		node_types = get_node_types_from_action_counts(self.action_counts)
		with self.test_session() as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(node_types), len(expected_node_types))
			for i in range(len(node_types)):
				tf.assert_equal(node_types[i], expected_node_types[i])

	def test_distribute_strategies_to_nodes(self):
		"""
		Test on `domains.hunger_games`

		The test `test_distribute_strategies_to_nodes` failed work on the CPU version of TensorFlow due to different
		 behavior of `tf.gather` on GPU and CPU:

		> Note that on CPU, if an out of bound index is found, an error is returned. On GPU, if an out of bound index is
		 found, a 0 is stored in the corresponding output value.

		(quoted from https://www.tensorflow.org/api_docs/python/tf/gather)

		TODO test also for different values of `updating_player` and `acting_players`
		"""
		inner_node_to_infoset = [
			tf.boolean_mask(
				indices,
				mask=self.mask_of_inner_nodes[level],
				name="non_terminal_infoset_strategies_lvl{}".format(level)
			)
			for level, indices in enumerate(self.node_to_infoset)
		]
		nodal_strategies = [
			distribute_strategies_to_nodes(
				self.infoset_strategies[level],
				inner_node_to_infoset[level],
				"nodal_strategies_lvl{}".format(level)
			)
			for level in range(len(self.infoset_strategies))
		]

		with self.test_session(
			# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to test on CPUs
		) as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(nodal_strategies), len(self.expected_nodal_strategies))
			for level, nodal_strategy in enumerate(nodal_strategies):
				print("\n>>>>>>>>>>>>>>>>>>Level {}<<<<<<<<<<<<<<<<<<".format(level))
				print_tensors(sess, [
					self.infoset_strategies[level],
					self.mask_of_inner_nodes[level],
					self.node_to_infoset[level],
					inner_node_to_infoset[level],
					nodal_strategy
				])
				print("\nexpected_nodal_strategies[{}]:".format(level))
				pprint(self.expected_nodal_strategies[level], width=50)
				np.testing.assert_array_almost_equal(
					sess.run(nodal_strategy),
					self.expected_nodal_strategies[level],
					err_msg="Nodal strategies differ at level {}!".format(level)
				)

	def test_distribute_strategies_to_inner_nodes(self):
		"""
		Test on `domains.hunger_games`

		The test `test_distribute_strategies_to_nodes` failed work on the CPU version of TensorFlow due to different
		 behavior of `tf.gather` on GPU and CPU:

		> Note that on CPU, if an out of bound index is found, an error is returned. On GPU, if an out of bound index is
		 found, a 0 is stored in the corresponding output value.

		(quoted from https://www.tensorflow.org/api_docs/python/tf/gather)

		TODO test also for different values of `updating_player` and `acting_players`
		"""
		nodal_strategies = [
			distribute_strategies_to_inner_nodes(
				self.infoset_strategies[level],
				self.node_to_infoset[level],
				self.mask_of_inner_nodes[level],
				"nodal_strategies_lvl{}".format(level)
			)
			for level in range(len(self.infoset_strategies))
		]

		with self.test_session(
			# config=tf.ConfigProto(device_count={'GPU': 0})  # uncomment to test on CPUs
		) as sess:
			sess.run(tf.global_variables_initializer())
			self.assertEquals(len(nodal_strategies), len(self.expected_nodal_strategies))
			for level, nodal_strategy in enumerate(nodal_strategies):
				print("\n>>>>>>>>>>>>>>>>>>Level {}<<<<<<<<<<<<<<<<<<".format(level))
				print_tensors(sess, [
					self.infoset_strategies[level],
					self.mask_of_inner_nodes[level],
					self.node_to_infoset[level],
					nodal_strategy
				])
				print("\nexpected_nodal_strategies[{}]:".format(level))
				pprint(self.expected_nodal_strategies[level], width=50)
				np.testing.assert_array_almost_equal(
					sess.run(nodal_strategy),
					self.expected_nodal_strategies[level],
					err_msg="Nodal strategies differ at level {}!".format(level)
				)
