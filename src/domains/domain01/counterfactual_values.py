#!/usr/bin/env python3

import tensorflow as tf

from domains.domain01.bottomup_expected_values import get_expected_values
from domains.domain01.domain_01 import levels, IS_strategies_lvl0, IS_strategies_lvl1, IS_strategies_lvl2, \
	node_to_IS_lvl0, node_to_IS_lvl1, node_to_IS_lvl2
from domains.domain01.topdown_reach_probabilities import get_reach_probabilities
from utils.tensor_utils import print_tensors


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def get_cf_values_nodes():  # TODO verify and write a unittest
	expected_values = get_expected_values()
	reach_probabilities = get_reach_probabilities()
	cf_values_of_nodes = [tf.multiply(reach_probabilities[level], expected_values[level],
	                                  name="node_cf_val_lvl{}".format(level)) for level in range(levels)]
	return cf_values_of_nodes


# noinspection PyPep8Naming
def get_cf_values_IS_actions():  # TODO verify and write a unittest
	node_to_IS = [node_to_IS_lvl0, node_to_IS_lvl1, node_to_IS_lvl2]
	node_cf_values = get_cf_values_nodes()
	IS_strategies = [IS_strategies_lvl0, IS_strategies_lvl1, IS_strategies_lvl2]
	cf_values_IS_actions = [None] * (levels - 1)
	cf_values_IS_actions[0] = tf.expand_dims(node_cf_values[1], axis=0, name="cf_values_IS_action_lvl0")
	for level in range(1, levels - 1):  # TODO replace for-loop with parallel_map on TensorArray?
		# cf_values_IS_actions[level] = tf.scatter_nd_add(ref=tf.Variable(tf.zeros_like(IS_strategies[level])),
		#                                                 indices=tf.expand_dims(node_to_IS[level], axis=-1),
		#                                                 updates=node_cf_values[level + 1],
		#                                                 name="cf_values_IS_action_lvl{}".format(level))
		# TODO the following is a hack, since `scatter_nd` resolves duplicate indices by cumulating! Find a proper solution!
		cf_values_IS_actions[level] = tf.scatter_nd(indices=tf.expand_dims(node_to_IS[level], axis=-1),
		                                            updates=node_cf_values[level + 1], shape=IS_strategies[level].shape,
		                                            name="cf_values_IS_action_lvl{}".format(level))
	return cf_values_IS_actions


# noinspection PyPep8Naming
def get_cf_values_IS():  # TODO verify and write a unittest
	IS_strategies = [IS_strategies_lvl0, IS_strategies_lvl1, IS_strategies_lvl2]
	cf_values_IS_actions = get_cf_values_IS_actions()
	return [tf.expand_dims(tf.reduce_sum(IS_strategies[level] * cf_values_IS_actions[level], axis=-1), axis=-1,
	                       name="cf_values_IS_lvl{}".format(level))
	        for level in range(levels - 1)]


if __name__ == '__main__':
	reach_probabilities_ = get_reach_probabilities()
	expected_values_ = get_expected_values()
	cf_values_nodes_ = get_cf_values_nodes()
	IS_strategies_ = [IS_strategies_lvl0, IS_strategies_lvl1, IS_strategies_lvl2]
	cf_values_IS_actions_ = get_cf_values_IS_actions()
	cf_values_IS_ = get_cf_values_IS()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [reach_probabilities_[i], expected_values_[i], cf_values_nodes_[i]])
			if i < levels - 1:
				print_tensors(sess, [IS_strategies_[i], cf_values_nodes_[i], cf_values_IS_actions_[i], cf_values_IS_[i]])
				# TODO unittest for multiple call of `cf_values_IS` and `cf_values_IS_actions` as below:
				# print_tensors(sess, [cf_values_IS_actions_[i], cf_values_IS_[i], cf_values_IS_actions_[i], cf_values_IS_[i]])
