#!/usr/bin/env python3

import tensorflow as tf

from src.domains.matching_pennies.counterfactual_values import assign_new_cf_values_infoset_actions, \
	get_cf_values_infoset
from src.domains.matching_pennies.domain_definitions import levels, positive_cumulative_regrets
from src.utils.tensor_utils import print_tensors


# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

def get_regrets():  # TODO verify and write a unittest
		cf_values_infoset_actions = assign_new_cf_values_infoset_actions()
		cf_values_infoset = get_cf_values_infoset()
		return [tf.subtract(cf_values_infoset_actions[level], cf_values_infoset[level], name="regrets_lvl{}".format(level))
						for level in range(levels - 1)]


def update_positive_cumulative_regrets(regrets=get_regrets()):  # TODO verify and write a unittest
	updated_regrets = [None] * (levels - 1)
	for level in range(levels - 1):
		# to keep cumulative regret still positive:
		maximum_addition = tf.maximum(
			- positive_cumulative_regrets[level],
			regrets[level]
		)
		updated_regrets[level] = tf.assign_add(
			ref=positive_cumulative_regrets[level],
			value=maximum_addition,
			name="update_regrets_lvl{}".format(level)
		)
	return updated_regrets


if __name__ == '__main__':
	cf_values_infoset_actions_ = assign_new_cf_values_infoset_actions()
	cf_values_infoset_ = get_cf_values_infoset()
	regrets_ = get_regrets()
	update_regrets_ops = update_positive_cumulative_regrets()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [cf_values_infoset_actions_[i], cf_values_infoset_[i], regrets_[i]])
			# TODO create a unit out of the following `print_tensors()`
			print_tensors(sess, [positive_cumulative_regrets[i],
			                     positive_cumulative_regrets[i],
			                     update_regrets_ops[i],
			                     positive_cumulative_regrets[i],
			                     positive_cumulative_regrets[i],
			                     update_regrets_ops[i],
			                     positive_cumulative_regrets[i],
			                     positive_cumulative_regrets[i]])
