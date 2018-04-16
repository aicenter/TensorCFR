#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_matching_pennies.counterfactual_values import assign_new_cf_values_infoset_actions, \
	get_cf_values_infoset
from src.domains.matching_pennies.domain_definitions import levels, positive_cumulative_regrets,\
	infoset_acting_players, current_updating_player, acting_depth
from src.utils.tensor_utils import print_tensors, masked_assign


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
		# TODO optimize by: pre-define `infosets_of_player1` and `infosets_of_player2` (in domain definitions) and switch
		infosets_of_updating_player = tf.reshape(
				tf.equal(infoset_acting_players[level], current_updating_player),
				shape=[positive_cumulative_regrets[level].shape[0]],
				name="infosets_of_updating_player_lvl{}".format(level)
		)
		# TODO implement and use `masked_assign_add` here
		updated_regrets[level] = masked_assign(
			ref=positive_cumulative_regrets[level],
			mask=infosets_of_updating_player,
			value=positive_cumulative_regrets[level] + maximum_addition,
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

		for level_ in range(acting_depth):
			print("########## Level {} ##########".format(level_))
			print_tensors(sess, [cf_values_infoset_actions_[level_], cf_values_infoset_[level_], regrets_[level_]])
			print("___________________________________\n")
			# TODO create a unit out of the following `print_tensors()`
			print_tensors(sess, [
				infoset_acting_players[level_],
				current_updating_player,
				positive_cumulative_regrets[level_],
				positive_cumulative_regrets[level_],
				update_regrets_ops[level_],
				positive_cumulative_regrets[level_],
				positive_cumulative_regrets[level_],
				update_regrets_ops[level_],
				positive_cumulative_regrets[level_],
				positive_cumulative_regrets[level_]
			])
