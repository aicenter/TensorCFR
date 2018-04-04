#!/usr/bin/env python3

import tensorflow as tf

from domains.domain01.counterfactual_values import get_cf_values_IS_actions, get_cf_values_IS
from domains.domain01.domain_01 import levels, positive_cumulative_regrets
from utils.tensor_utils import print_tensors


# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png

def get_regrets():  # TODO verify and write a unittest
	cf_values_IS_actions = get_cf_values_IS_actions()
	cf_values_IS = get_cf_values_IS()
	return [tf.subtract(cf_values_IS_actions[level], cf_values_IS[level], name="regrets_lvl{}".format(level))
	        for level in range(levels - 1)]


def update_positive_cumulative_regrets(regrets=get_regrets()):  # TODO verify and write a unittest
	updated_regrets = [None] * (levels - 1)
	for level in range(levels - 1):
		maximum_addition = tf.maximum(- positive_cumulative_regrets[level],  # to keep cumulative regret still positive
		                              regrets[level])
		updated_regrets[level] = tf.assign_add(ref=positive_cumulative_regrets[level], value=maximum_addition,
		                                       name="update_regrets_lvl{}".format(level))
	return updated_regrets


if __name__ == '__main__':
	cf_values_IS_actions_ = get_cf_values_IS_actions()
	cf_values_IS_ = get_cf_values_IS()
	regrets_ = get_regrets()
	update_regrets = update_positive_cumulative_regrets()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for i in range(levels - 1):
			print("########## Level {} ##########".format(i))
			print_tensors(sess, [cf_values_IS_actions_[i], cf_values_IS_[i], regrets_[i], update_regrets[i],
			                     positive_cumulative_regrets[i]])

			# TODO create a unit out of following lines
			# print_tensors(sess, [regrets_[i]])
			# print_tensors(sess, [positive_cumulative_regrets[i]])
			# sess.run(update_regrets[i])
			# print_tensors(sess, [positive_cumulative_regrets[i]])
			# print_tensors(sess, [update_regrets[i]])
			# print_tensors(sess, [positive_cumulative_regrets[i], positive_cumulative_regrets[i]])
