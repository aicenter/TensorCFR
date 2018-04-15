#!/usr/bin/env python3

import tensorflow as tf

from src.constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, CHANCE_PLAYER, PLAYER1, \
	PLAYER2, DEFAULT_AVERAGING_DELAY
from src.utils.tensor_utils import print_tensors

# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

actions_per_levels = [2, 2]
levels = len(actions_per_levels) + 1
acting_depth = len(actions_per_levels)

# Init
node_to_infoset = [None] * acting_depth
shape = [None] * levels  # TODO replace with: shape = [actions_per_levels[:i] for i in range(levels)]
node_types = [None] * levels
utilities = [None] * levels
infoset_acting_players = [None] * acting_depth
current_infoset_strategies = [None] * acting_depth

########## Level 0 ##########
node_to_infoset[0] = tf.Variable(0, name="node_to_infoset_lvl0")
reach_probability_of_root_node = tf.Variable(1.0, name="reach_probability_of_root_node")
shape[0] = actions_per_levels[:0]
node_types[0] = tf.Variable(INNER_NODE, name="node_types_lvl0")
utilities[0] = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[0], name="utilities_lvl0")
infoset_acting_players[0] = tf.Variable([PLAYER1], name="infoset_acting_players_lvl0")
current_infoset_strategies[0] = tf.Variable([[0.1, 0.9]], name="current_infoset_strategies_lvl0")

########## Level 1 ##########
node_to_infoset[1] = tf.Variable([0, 0], name="node_to_infoset_lvl1")
shape[1] = actions_per_levels[:1]
node_types[1] = tf.Variable([INNER_NODE] * 2, name="node_types_lvl1")
utilities[1] = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[1], name="utilities_lvl1")
infoset_acting_players[1] = tf.Variable([PLAYER2], name="infoset_acting_players_lvl1")
current_infoset_strategies[1] = tf.Variable([[0.2, 0.8]], name="current_infoset_strategies_lvl1")

########## Level 2 ##########
# There are never any infosets in the final layer, only terminal / imaginary nodes.
shape[2] = actions_per_levels[:2]
node_types[2] = tf.Variable(tf.fill(value=TERMINAL_NODE, dims=shape[2]), name="node_types_lvl2")
utilities[2] = tf.Variable(
		[
			[-1.0, 1.0],
			[1.0, -1.0]
		],
		name="utilities_lvl2"
)

########## miscellaneous tensors ##########
cf_values_infoset_actions = [
	# TODO remove and compute again for every iteration
	tf.Variable(tf.zeros_like(current_infoset_strategies[0]), name="cf_values_infoset_actions_lvl0"),
	tf.Variable(tf.zeros_like(current_infoset_strategies[1]), name="cf_values_infoset_actions_lvl1"),
]
positive_cumulative_regrets = [
	# TODO rewrite to list comprehension
	tf.Variable(tf.zeros_like(current_infoset_strategies[0]), name="positive_cumulative_regrets_lvl0"),
	tf.Variable(tf.zeros_like(current_infoset_strategies[1]), name="positive_cumulative_regrets_lvl1"),
]
cumulative_infoset_strategies = [tf.Variable(tf.zeros_like(current_infoset_strategies[level]),
                                             name="cumulative_infoset_strategies_lvl{}".format(level))
                                 for level in range(acting_depth)]  # used for the final average strategy
infosets_of_non_chance_player = [
	tf.reshape(tf.not_equal(infoset_acting_players[level], CHANCE_PLAYER),
	           shape=[current_infoset_strategies[level].shape[0]],
	           name="infosets_of_acting_player_lvl{}".format(level))
	for level in range(acting_depth)
]
cfr_step = tf.Variable(initial_value=0, dtype=tf.int64, name="cfr_step")  # counter of CFR+ iterations
averaging_delay = tf.constant(  # https://arxiv.org/pdf/1407.5042.pdf (Figure 2)
		DEFAULT_AVERAGING_DELAY,
		dtype=cfr_step.dtype,
		name="averaging_delay"
)
player_owning_the_utilities = tf.constant(PLAYER1, name="player_owning_the_utilities")  # utilities defined from this...
#  ...player's point of view
current_updating_player = tf.Variable(initial_value=PLAYER1, name="current_updating_player")
current_opponent = tf.Variable(initial_value=PLAYER2, name="current_opponent")
signum_of_current_player = tf.where(
		condition=tf.equal(current_updating_player, player_owning_the_utilities),
		x=1.0,
		y=-1.0,  # Opponent's utilities in zero-sum games = (-utilities) of `player_owning_the_utilities`
		name="signum_of_current_player",
)


def get_node_types():
	return node_types


def get_node_to_infoset():
	return node_to_infoset


def get_infoset_acting_players():
	return infoset_acting_players


def print_player_variables(session):
	print("########## Misc ##########")
	print_tensors(session, [
		cfr_step,
		current_updating_player,
		current_opponent,
		signum_of_current_player,
		player_owning_the_utilities,
	])


if __name__ == '__main__':
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		levels_range = range(levels)
		for level in levels_range:
			print("########## Level {} ##########".format(level))
			if level == 0:
				print_tensors(sess, [reach_probability_of_root_node])
			print_tensors(sess, [node_types[level], utilities[level]])
			if level != levels_range[-1]:
				print_tensors(sess, [
					node_to_infoset[level],
					infoset_acting_players[level],
					current_infoset_strategies[level],
					cf_values_infoset_actions[level],
					positive_cumulative_regrets[level],
					cumulative_infoset_strategies[level],
				])
		print_player_variables(session=sess)
