#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr_domain01.node_strategies import get_node_strategies
from src.commons.constants import TERMINAL_NODE
from src.domains.domain01.domain_definitions import utilities, node_types, levels, signum_of_current_player, \
	print_misc_variables, current_infoset_strategies, current_updating_player
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_expected_values():
	node_strategies = get_node_strategies()
	with tf.variable_scope("expected_values"):
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


def get_infoset_cbr_support(for_player=current_updating_player, strategies=current_infoset_strategies):
	"""
	Compute the support of infosets' counterfactual best response (CBR) values for player `for_player` given opponent's
	 strategies `strategies`. See "Solving Imperfect Information Games Using Decomposition"
	  https://arxiv.org/pdf/1303.4441.pdf

	:param for_player: The player for which the support computed. Set to `current_updating_player` by default.
	:param strategies: List of tensors representing infosets' strategies, for which CBR support is being computed.
	:return: The list of boolean tensors, each containing the support of infosets's CBR at each level.
	"""
	# TODO continue here
	pass


def get_infoset_cbr_values(for_player=current_updating_player, strategies=current_infoset_strategies):
	"""
	Compute infosets' counterfactual best response (CBR) values for player `for_player` given the strategies `strategies`.
	See "Solving Imperfect Information Games Using Decomposition" https://arxiv.org/pdf/1303.4441.pdf

	:param for_player: The player for which the counterfactual best response values are computed. Set to
	 `current_updating_player` by default.
	:param strategies: List of tensors representing infosets' strategies, for which CBR values are being computed.
	:return: The list of tensors, each containing counterfactual best response of infosets at each level.
	"""
	# TODO continue here
	pass
	node_strategies = get_node_strategies()
	with tf.variable_scope("expected_values"):
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
	print_misc_variables(session=session)
	for level_ in reversed(range(levels)):
		print("########## Level {} ##########".format(level_))
		if level_ < len(node_strategies_):
			print_tensors(session, [node_strategies_[level_]])
		print_tensors(session, [
			tf.multiply(
					signum_of_current_player,
					utilities[level_],
					name="signum_utilities_lvl{}".format(level_)
			),
			expected_values_[level_]
		])


if __name__ == '__main__':
	from src.algorithms.tensorcfr_domain01.swap_players import swap_players
	node_strategies_ = get_node_strategies()
	expected_values_ = get_expected_values()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# TODO extract following lines to a UnitTest
		show_expected_values(sess)
		sess.run(swap_players())
		print("-----------Swap players-----------\n")
		show_expected_values(sess)
