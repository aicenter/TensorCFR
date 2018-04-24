#!/usr/bin/env python3

import tensorflow as tf

from src.commons.constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, IMAGINARY_NODE, CHANCE_PLAYER, \
	PLAYER1, \
	PLAYER2, NO_ACTING_PLAYER, DEFAULT_AVERAGING_DELAY, INT_DTYPE
from src.utils.tensor_utils import print_tensors

# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

with tf.variable_scope("domain_definitions"):
	actions_per_levels = [5, 3, 2]  # maximum number of actions per each level (0, 1, 2)
	levels = len(actions_per_levels) + 1  # accounting for 0th level
	acting_depth = len(actions_per_levels)

	# Init
	node_to_infoset = [None] * acting_depth
	shape = [None] * levels  # TODO replace with: shape = [actions_per_levels[:i] for i in range(levels)]
	node_types = [None] * levels
	utilities = [None] * levels
	infoset_acting_players = [None] * acting_depth
	initial_infoset_strategies = [None] * acting_depth

	########## Level 0 ##########
	# I0,0 = { s } ... root node, the chance player acts here
	# there are 5 actions in node s
	node_to_infoset[0] = tf.get_variable("node_to_infoset_lvl0", initializer=0)
	reach_probability_of_root_node = tf.get_variable("reach_probability_of_root_node", initializer=1.0)
	shape[0] = actions_per_levels[:0]
	node_types[0] = tf.get_variable("node_types_lvl0", initializer=INNER_NODE)
	utilities[0] = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[0], name="utilities_lvl0")
	infoset_acting_players[0] = tf.get_variable("infoset_acting_players_lvl0", initializer=[CHANCE_PLAYER])
	initial_infoset_strategies[0] = tf.placeholder_with_default(
			input=[[0.5, .25, 0.1, 0.1, .05]],  # of I0,0
			shape=[infoset_acting_players[0].shape[0], actions_per_levels[0]],
			name="initial_infoset_strategies_lvl{}".format(0)
	)

	########## Level 1 ##########
	# I1,0 = { s' }
	# I1,1 = { s1 }
	# I1,2 = { s2, s3 }
	# I1,3 = { s4 } ... chance node
	# each node has 3 actions
	node_to_infoset[1] = tf.get_variable("node_to_infoset_lvl1", initializer=[0, 1, 2, 2, 3])
	shape[1] = actions_per_levels[:1]
	node_types[1] = tf.get_variable("node_types_lvl1", initializer=[INNER_NODE] * 5)
	utilities[1] = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[1], name="utilities_lvl1")
	infoset_acting_players[1] = tf.get_variable(
			"infoset_acting_players_lvl1",
			initializer=[PLAYER1,         # I1,0
			             PLAYER2,         # I1,1
			             PLAYER2,         # I1,2
			             CHANCE_PLAYER],  # I1,3
	)
	initial_infoset_strategies[1] = tf.placeholder_with_default(
			input=[[0.5, 0.4, 0.1],   # of I1,0
						 [0.1, 0.9, 0.0],   # of I1,1
						 [0.2, 0.8, 0.0],   # of I1,2
						 [0.3, 0.3, 0.3]],  # of I1,3
			shape=[infoset_acting_players[1].shape[0], actions_per_levels[1]],
			name="initial_infoset_strategies_lvl{}".format(1)
	)

	########## Level 2 ##########
	# I2,0 = { s5 }
	# I2,1 = { s6 }
	# I2,2 = { s8, s9 }
	# I2,3 = { s11, s14 }
	# I2,4 = { s12, s15 }
	# I2,5 = { s18 }
	# I2,6 = { s19 }
	# I2,7 = { s7, s17 } ... terminal nodes
	# I2,8 = { s10, s13, s16 } ... imaginary nodes
	# each node has 2 actions
	node_to_infoset[2] = tf.get_variable(
			"node_to_infoset_lvl2",
			initializer=[[0, 1, 7],   # s5, s6, s7
			             [2, 2, 8],   # s8, s9, s10
			             [3, 4, 8],   # s11, s12, s1
			             [3, 4, 8],   # s14, s15, s16
			             [7, 5, 6]],  # s17, s18, s19
	)
	shape[2] = actions_per_levels[:2]
	node_types[2] = tf.get_variable(
			"node_types_lvl2",
			initializer=[[INNER_NODE, INNER_NODE, TERMINAL_NODE],   # s5, s6, s7
			             [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s8, s9, s10
			             [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s11, s12, s13
			             [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s14, s15, s16
			             [TERMINAL_NODE, INNER_NODE, INNER_NODE]],  # s17, s18, s19
	)
	utilities[2] = tf.get_variable(
			"utilities_lvl2",
			initializer=tf.where(
					condition=tf.equal(node_types[2], TERMINAL_NODE),
					x=tf.reshape(tf.range(10, 160, delta=10.0), shape[2]),
					y=tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[2]),
			),
	)
	infoset_acting_players[2] = tf.get_variable(
			"infoset_acting_players_lvl2",
			initializer=[PLAYER1,            # of I2,0
			             PLAYER2,            # of I2,1
			             PLAYER1,            # of I2,2
			             PLAYER2,            # of I2,3
			             CHANCE_PLAYER,      # of I2,4
			             PLAYER1,            # of I2,5
			             PLAYER2,            # of I2,6
			             NO_ACTING_PLAYER,   # of I2,7 ... pseudo-infoset of terminal nodes
			             NO_ACTING_PLAYER],  # of I2,8 ... pseudo-infoset of imaginary nodes
	)
	initial_infoset_strategies[2] = tf.placeholder_with_default(
			input=[[0.15, 0.85],   # of I2,0
			       [0.70, 0.30],   # of I2,1
			       [0.25, 0.75],   # of I2,2
			       [0.50, 0.50],   # of I2,3
			       [0.10, 0.90],   # of I2,4
			       [0.45, 0.55],   # of I2,5
			       [0.40, 0.60],   # of I2,6
			       [0.00, 0.00],   # of I2,7 ... terminal nodes <- mock-up zero strategy
			       [0.00, 0.00]],  # of I2,8 ... imaginary nodes <- mock-up zero strategy
			shape=[infoset_acting_players[2].shape[0], actions_per_levels[2]],
			name="initial_infoset_strategies_lvl{}".format(2)
	)

	########## Level 3 ##########
	# There are never any infosets in the final layer, only terminal / imaginary nodes.
	shape[3] = actions_per_levels[:3]
	indices_of_imaginary_nodes_lvl3 = tf.constant(
		[[0, 2],   # children of s7
		 [1, 2],   # children of s10
		 [2, 2],   # children of s13
		 [3, 2],   # children of s16
		 [4, 0]],  # children of s17
		name="indices_of_imaginary_nodes_lvl3"
	)
	node_types[3] = tf.get_variable(
			"node_types_lvl3",
			initializer=
			[[[1, 1],
			  [1, 1],
			  [2, 2]],

			 [[1, 1],
			  [1, 1],
			  [2, 2]],

			 [[1, 1],
			  [1, 1],
			  [2, 2]],

			 [[1, 1],
			  [1, 1],
			  [2, 2]],

			 [[2, 2],
			  [1, 1],
			  [1, 1]]]
	)
	utilities[3] = tf.get_variable(
			"utilities_lvl3",
			initializer=tf.where(
					condition=tf.equal(node_types[3], TERMINAL_NODE),
					x=tf.reshape(tf.range(10, 310, delta=10.0), shape[3]),
					y=tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[3]),
			),
	)

	########## miscellaneous tensors ##########
	current_infoset_strategies = [
		tf.get_variable(
				"current_infoset_strategies_lvl{}".format(level),
				initializer=initial_infoset_strategies[level],
		) for level in range(acting_depth)
	]
	positive_cumulative_regrets = [
		tf.get_variable(
				"positive_cumulative_regrets_lvl{}".format(level),
				initializer=tf.zeros_like(current_infoset_strategies[level]),
		) for level in range(acting_depth)
	]
	cumulative_infoset_strategies = [
		tf.get_variable(
				"cumulative_infoset_strategies_lvl{}".format(level),
				initializer=tf.zeros_like(current_infoset_strategies[level]),
		) for level in range(acting_depth)
	]  # used for the final average strategy
	infosets_of_non_chance_player = [
		tf.reshape(
				tf.not_equal(infoset_acting_players[level], CHANCE_PLAYER),
				shape=[current_infoset_strategies[level].shape[0]],
				name="infosets_of_acting_player_lvl{}".format(level)
		) for level in range(acting_depth)
	]
	cfr_step = tf.get_variable(  # counter of CFR+ iterations
			"cfr_step",
			initializer=1,
			dtype=INT_DTYPE,
	)
	averaging_delay = tf.get_variable(  # https://arxiv.org/pdf/1407.5042.pdf (Figure 2)
			"averaging_delay",
			initializer=DEFAULT_AVERAGING_DELAY,
			dtype=INT_DTYPE,
	)
	player_owning_the_utilities = tf.constant(  # utilities defined...
			PLAYER1,  # ...from this player's point of view
			name="player_owning_the_utilities"
	)
	current_updating_player = tf.get_variable(
			"current_updating_player",
			initializer=PLAYER1,
			dtype=INT_DTYPE,
	)
	current_opponent = tf.get_variable(
			"current_opponent",
			initializer=PLAYER2,
			dtype=INT_DTYPE,
	)
	signum_of_current_player = tf.where(
			condition=tf.equal(current_updating_player, player_owning_the_utilities),
			x=1.0,
			y=-1.0,  # Opponent's utilities in zero-sum games = (-utilities) of `player_owning_the_utilities`
			name="signum_of_current_player",
	)


def get_node_types():
	with tf.variable_scope("domain_definitions", reuse=True):
		return [
			tf.get_variable("node_types_lvl{}".format(level), dtype=INT_DTYPE)
			for level in range(levels)
		]


def get_infoset_acting_players():
	return infoset_acting_players


def print_misc_variables(session):
	print("########## Misc ##########")
	print_tensors(session, [
		cfr_step,
		averaging_delay,
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
			print_tensors(sess, [get_node_types()[level], utilities[level]])
			if level != levels_range[-1]:
				print_tensors(sess, [
					node_to_infoset[level],
					infoset_acting_players[level],
					initial_infoset_strategies[level],
					current_infoset_strategies[level],
					positive_cumulative_regrets[level],
					cumulative_infoset_strategies[level],
				])
		print_misc_variables(session=sess)
