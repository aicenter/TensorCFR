#!/usr/bin/env python3

import tensorflow as tf

from src.constants import TERMINAL_NODE
from src.domains.domain01.domain01 import node_to_infoset, infoset_strategies, utilities, node_types, levels
from src.utils.assign_strategies_to_nodes import assign_strategies_to_nodes
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png


def get_node_strategies():
	node_strategies = [None] * (levels - 1)
	for level in range(levels - 1):
		node_strategies[level] = assign_strategies_to_nodes(
				infoset_strategies[level],
				node_to_infoset[level],
				name="node_strategies_lvl{}".format(level))
	return node_strategies


def get_expected_values():
	node_strategies = get_node_strategies()
	expected_values = [None] * levels
	expected_values[levels - 1] = tf.identity(utilities[levels - 1], name="expected_values_lvl{}".format(levels - 1))
	for level in reversed(range(levels - 1)):
		weighted_sum_of_values = tf.reduce_sum(
				input_tensor=node_strategies[level] * expected_values[level + 1],
				axis=-1,
				name="weighted_sum_of_values_lvl{}".format(level))
		expected_values[level] = tf.where(
				condition=tf.equal(node_types[level], TERMINAL_NODE),
				x=utilities[level],
				y=weighted_sum_of_values,
				name="expected_values_lvl{}".format(level))
	return expected_values


if __name__ == '__main__':
	node_strategies_ = get_node_strategies()
	expected_values_ = get_expected_values()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for lvl in reversed(range(levels)):
			print("########## Level {} ##########".format(lvl))
			if lvl < len(node_strategies_):
				print_tensors(sess, [node_strategies_[lvl]])
			print_tensors(sess, [utilities[lvl], expected_values_[lvl]])
