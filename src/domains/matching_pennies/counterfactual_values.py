#!/usr/bin/env python3

import tensorflow as tf

from src.domains.matching_pennies.bottomup_expected_values import get_expected_values
from src.domains.matching_pennies.domain_definitions import levels, current_infoset_strategies, node_to_infoset, \
	cf_values_infoset_actions
from src.domains.matching_pennies.topdown_reach_probabilities import get_nodal_reach_probabilities
from src.utils.tensor_utils import print_tensors, scatter_nd_sum


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_cf_values_nodes():  # TODO verify and write a unittest
	expected_values = get_expected_values()
	reach_probabilities = get_nodal_reach_probabilities()
	cf_values_of_nodes = [tf.multiply(reach_probabilities[level], expected_values[level],
	                                  name="node_cf_val_lvl{}".format(level)) for level in range(levels)]
	return cf_values_of_nodes


def assign_new_cf_values_infoset_actions():  # TODO verify and write a unittest
	node_cf_values = get_cf_values_nodes()
	new_cf_values_infoset_action = [None] * (levels - 1)
	new_cf_values_infoset_action[0] = tf.assign(
			ref=cf_values_infoset_actions[0],
			value=tf.expand_dims(node_cf_values[1], axis=0)
	)
	for level in range(1, levels - 1):  # TODO replace for-loop with parallel_map on TensorArray?
		new_cf_values_infoset_action[level] = scatter_nd_sum(
				indices=tf.expand_dims(node_to_infoset[level], axis=-1),
				updates=node_cf_values[level + 1],
				shape=current_infoset_strategies[level].shape,
		)
	return [tf.assign(ref=cf_values_infoset_actions[level], value=new_cf_values_infoset_action[level],
	                  name="assign_new_cfv_infoset_action_lvl{}".format(level)) for level in range(levels - 1)]


def get_cf_values_infoset():  # TODO verify and write a unittest
	return [tf.expand_dims(tf.reduce_sum(current_infoset_strategies[level] * cf_values_infoset_actions[level], axis=-1),
	                       axis=-1, name="cf_values_infoset_lvl{}".format(level))
	        for level in range(levels - 1)]


if __name__ == '__main__':
	nodal_reach_probabilities_ = get_nodal_reach_probabilities()
	expected_values_ = get_expected_values()
	cf_values_nodes_ = get_cf_values_nodes()
	infoset_strategies_ = current_infoset_strategies
	assign_new_cf_values_infoset_actions_ = assign_new_cf_values_infoset_actions()
	cf_values_infoset_ = get_cf_values_infoset()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [nodal_reach_probabilities_[i], expected_values_[i], cf_values_nodes_[i]])
			if i < levels - 1:
				print_tensors(sess, [infoset_strategies_[i], assign_new_cf_values_infoset_actions_[i],
				                     cf_values_infoset_actions[i], cf_values_infoset_[i]])
				# TODO unittest for multiple call of `cf_values_infoset` and `cf_values_infoset_actions` as below:
				print_tensors(sess, [
					cf_values_infoset_actions[i],
					cf_values_infoset_actions[i],
					assign_new_cf_values_infoset_actions_[i],
					cf_values_infoset_actions[i],
					cf_values_infoset_actions[i],
					assign_new_cf_values_infoset_actions_[i],
					cf_values_infoset_actions[i],
					cf_values_infoset_actions[i],
				])
