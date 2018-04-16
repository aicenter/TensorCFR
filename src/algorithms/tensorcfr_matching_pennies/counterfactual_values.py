#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_matching_pennies.topdown_reach_probabilities import get_nodal_reach_probabilities
from src.algorithms.tensorcfr_matching_pennies.bottomup_expected_values import get_expected_values
from src.domains.matching_pennies.domain_definitions import levels, current_infoset_strategies, node_to_infoset
from src.utils.tensor_utils import print_tensors, scatter_nd_sum


# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

def get_cf_values_nodes():  # TODO verify and write a unittest
	expected_values = get_expected_values()
	reach_probabilities = get_nodal_reach_probabilities()
	cf_values_of_nodes = [tf.multiply(reach_probabilities[level], expected_values[level],
	                                  name="node_cf_val_lvl{}".format(level)) for level in range(levels)]
	return cf_values_of_nodes


def get_cf_values_infoset_actions():  # TODO verify and write a unittest
	node_cf_values = get_cf_values_nodes()
	cf_values_infoset_actions = [None] * (levels - 1)
	cf_values_infoset_actions[0] = tf.expand_dims(
			node_cf_values[1],
			axis=0,
			name="cf_values_infoset_actions_lvl0")
	for level in range(1, levels - 1):  # TODO replace for-loop with parallel_map on TensorArray?
		cf_values_infoset_actions[level] = scatter_nd_sum(
				indices=tf.expand_dims(node_to_infoset[level], axis=-1),
				updates=node_cf_values[level + 1],
				shape=current_infoset_strategies[level].shape,
				name="cf_values_infoset_actions_lvl{}".format(level),
		)
	return cf_values_infoset_actions    # TODO remove the same tensor from `domain_definitions.py`


def get_cf_values_infoset():  # TODO verify and write a unittest
	cf_values_infoset_actions = get_cf_values_infoset_actions()
	return [
		tf.expand_dims(
				tf.reduce_sum(
						current_infoset_strategies[level] * cf_values_infoset_actions[level],
						axis=-1
				),
				axis=-1,
				name="cf_values_infoset_lvl{}".format(level)
		)
		for level in range(levels - 1)
	]


if __name__ == '__main__':
	nodal_reach_probabilities_ = get_nodal_reach_probabilities()
	expected_values_ = get_expected_values()
	cf_values_nodes_ = get_cf_values_nodes()
	cf_values_infoset_actions_ = get_cf_values_infoset_actions()
	cf_values_infoset_ = get_cf_values_infoset()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [
				nodal_reach_probabilities_[i],
				expected_values_[i],
				cf_values_nodes_[i]
			])
			if i < levels - 1:
				print_tensors(sess, [
					current_infoset_strategies[i],
					cf_values_infoset_actions_[i],
					cf_values_infoset_[i],
				])
