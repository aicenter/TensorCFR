#!/usr/bin/env python3

import tensorflow as tf

from src.commons.constants import TERMINAL_NODE
from src.domains.domain01.domain01 import node_to_infoset, current_infoset_strategies, utilities, node_types, levels, \
	signum_of_current_player, print_player_variables
from src.domains.domain01.node_strategies import assign_strategies_to_nodes
from src.domains.domain01.swap_players import swap_players
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_node_strategies():
	node_strategies = [None] * (levels - 1)
	for level in range(levels - 1):
		node_strategies[level] = assign_strategies_to_nodes(
				current_infoset_strategies[level],
				node_to_infoset[level],
				name="node_strategies_lvl{}".format(level)
		)
	return node_strategies


def get_expected_values():
	node_strategies = get_node_strategies()
	expected_values = [None] * levels
	expected_values[levels - 1] = tf.multiply(
			signum_of_current_player,
			utilities[levels - 1],
			name="expected_values_lvl{}".format(levels - 1)
	)
	for level in reversed(range(levels - 1)):
		weighted_sum_of_values = tf.reduce_sum(
				input_tensor=node_strategies[level] * expected_values[level + 1],
				axis=-1,
				name="weighted_sum_of_values_lvl{}".format(level))
		expected_values[level] = tf.where(
				condition=tf.equal(node_types[level], TERMINAL_NODE),
				x=signum_of_current_player * utilities[level],
				y=weighted_sum_of_values,
				name="expected_values_lvl{}".format(level))
	return expected_values


def show_expected_values(session):
	print_player_variables(session=session)
	for level_ in reversed(range(levels)):
		print("########## Level {} ##########".format(level_))
		if level_ < len(node_strategies_):
			print_tensors(session, [node_strategies_[level_]])
		print_tensors(session, [
			signum_of_current_player * utilities[level_],
			expected_values_[level_]
		])


if __name__ == '__main__':
	node_strategies_ = get_node_strategies()
	expected_values_ = get_expected_values()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# TODO extract following lines to a UnitTest
		show_expected_values(sess)
		sess.run(swap_players())
		print("-----------Swap players-----------\n")
		show_expected_values(sess)
