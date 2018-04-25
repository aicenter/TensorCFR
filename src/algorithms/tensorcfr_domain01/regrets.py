#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_domain01.counterfactual_values import get_infoset_cf_values_per_actions, \
	get_infoset_cf_values
from src.domains.domain01.domain_definitions import levels, positive_cumulative_regrets,\
	infoset_acting_players, current_updating_player, acting_depth
from src.utils.tensor_utils import print_tensors, masked_assign


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_regrets():  # TODO verify and write a unittest
	cf_values_infoset_actions = get_infoset_cf_values_per_actions()
	cf_values_infoset = get_infoset_cf_values()
	with tf.variable_scope("regrets"):
		return [
			tf.subtract(
					cf_values_infoset_actions[level],
					cf_values_infoset[level],
					name="regrets_lvl{}".format(level),
			) for level in range(levels - 1)
		]


def update_positive_cumulative_regrets(regrets=get_regrets()):  # TODO verify and write a unittest
	with tf.variable_scope("update_cumulative_regrets"):
		update_regrets_ops = [None] * acting_depth
		for level in range(acting_depth):
			with tf.variable_scope("update_cumulative_regrets_lvl{}".format(level)):
				# TODO optimize by: pre-define `infosets_of_player1` and `infosets_of_player2` (in domain definitions) and switch
				infosets_of_updating_player = tf.reshape(
						tf.equal(infoset_acting_players[level], current_updating_player),
						shape=[positive_cumulative_regrets[level].shape[0]],
						name="infosets_of_updating_player_lvl{}".format(level),
				)
				# TODO implement and use `masked_assign_add` here
				update_regrets_ops[level] = masked_assign(
					ref=positive_cumulative_regrets[level],
					mask=infosets_of_updating_player,
					value=tf.maximum(
							0.0,
							positive_cumulative_regrets[level] + regrets[level]
					),
					name="update_regrets_lvl{}".format(level)
				)
		return update_regrets_ops


if __name__ == '__main__':
	cf_values_infoset_actions_ = get_infoset_cf_values_per_actions()
	cf_values_infoset_ = get_infoset_cf_values()
	regrets_ = get_regrets()
	update_regrets_ops_ = update_positive_cumulative_regrets()
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
				update_regrets_ops_[level_],
				positive_cumulative_regrets[level_],
				positive_cumulative_regrets[level_],
				update_regrets_ops_[level_],
				positive_cumulative_regrets[level_],
				positive_cumulative_regrets[level_]
			])
