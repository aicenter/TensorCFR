#!/usr/bin/env python3

import tensorflow as tf

from src.constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, IMAGINARY_NODE, CHANCE_PLAYER, PLAYER1, \
	PLAYER2, NO_ACTING_PLAYER
from src.utils.tensor_utils import print_tensors, masked_assign

# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

actions_per_levels = [5, 3, 2]  # maximum number of actions per each level (0, 1, 2)
levels = len(actions_per_levels) + 1  # accounting for 0th level
acting_depth = len(actions_per_levels)

# Init
node_to_infoset = [None] * 3
shape = [None] * 4
node_types = [None] * 4
utilities = [None] * 4
infoset_acting_players = [None] * 3
current_infoset_strategies = [None] * 3


########## Level 0 ##########
# I0,0 = { s } ... root node, the chance player acts here
# there are 5 actions in node s
node_to_infoset[0] = tf.Variable(0, name="node_to_infoset_lvl0")
reach_probability_of_root_node = tf.Variable(1.0, name="reach_probability_of_root_node")
shape[0] = actions_per_levels[:0]
node_types[0] = tf.Variable(INNER_NODE, name="node_types_lvl0")
utilities[0] = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[0], name="utilities_lvl0")
infoset_acting_players[0] = tf.Variable(CHANCE_PLAYER, name="infoset_acting_players_lvl0")
current_infoset_strategies[0] = tf.Variable(
	[[0.5, .25, 0.1, 0.1, .05]],  # of I0,0
	name="current_infoset_strategies_lvl0"
)


########## Level 1 ##########
# I1,0 = { s' }
# I1,1 = { s1 }
# I1,2 = { s2, s3 }
# I1,3 = { s4 } ... chance node
# each node has 3 actions
node_to_infoset[1] = tf.Variable([0, 1, 2, 2, 3], name="node_to_infoset_lvl1")
shape[1] = actions_per_levels[:1]
node_types[1] = tf.Variable([INNER_NODE] * 5, name="node_types_lvl1")
utilities[1] = tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[1], name="utilities_lvl1")
infoset_acting_players[1] = tf.Variable(
	[PLAYER1,         # I1,0
	 PLAYER2,         # I1,1
	 PLAYER2,         # I1,2
	 CHANCE_PLAYER],  # I1,3
	name="infoset_acting_players_lvl1"
)
current_infoset_strategies[1] = tf.Variable(
	[[0.5, 0.4, 0.1],   # of I1,0
	 [0.1, 0.9, 0.0],   # of I1,1
	 [0.2, 0.8, 0.0],   # of I1,2
	 [0.3, 0.3, 0.3]],  # of I1,3
	name="current_infoset_strategies_lvl1"
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
node_to_infoset[2] = tf.Variable(
	[[0, 1, 7],   # s5, s6, s7
	 [2, 2, 8],   # s8, s9, s10
	 [3, 4, 8],   # s11, s12, s1
	 [3, 4, 8],   # s14, s15, s16
	 [7, 5, 6]],  # s17, s18, s19
	name="node_to_infoset_lvl2"
)
shape[2] = actions_per_levels[:2]
node_types[2] = tf.Variable(
	[[INNER_NODE, INNER_NODE, TERMINAL_NODE],   # s5, s6, s7
	 [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s8, s9, s10
	 [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s11, s12, s13
	 [INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s14, s15, s16
	 [TERMINAL_NODE, INNER_NODE, INNER_NODE]],  # s17, s18, s19
	name="node_types_lvl2"
)
utilities_lvl2_tmp = tf.Variable(tf.fill(value=NON_TERMINAL_UTILITY, dims=shape[2]))
mask_terminals_lvl2_tmp = tf.equal(node_types[2], TERMINAL_NODE)
terminal_values_lvl2_tmp = tf.reshape(tf.range(10, 160, delta=10.0), shape[2])
utilities_lvl2_tmp = masked_assign(
    ref=utilities_lvl2_tmp,
    value=terminal_values_lvl2_tmp,
    mask=mask_terminals_lvl2_tmp,
    name="utilities_lvl2"
)
utilities[2] = utilities_lvl2_tmp
infoset_acting_players[2] = tf.Variable(
	[PLAYER1,            # of I2,0
	 PLAYER2,            # of I2,1
	 PLAYER1,            # of I2,2
	 PLAYER2,            # of I2,3
	 CHANCE_PLAYER,      # of I2,4
	 PLAYER1,            # of I2,5
	 PLAYER2,            # of I2,6
	 NO_ACTING_PLAYER,   # of I2,7 ... pseudo-infoset of terminal nodes
	 NO_ACTING_PLAYER],  # of I2,8 ... pseudo-infoset of imaginary nodes
	name="infoset_acting_players_lvl2"
)
current_infoset_strategies[2] = tf.Variable(
	[[0.15, 0.85],   # of I2,0
	 [0.70, 0.30],   # of I2,1
	 [0.25, 0.75],   # of I2,2
	 [0.50, 0.50],   # of I2,3
	 [0.10, 0.90],   # of I2,4
	 [0.45, 0.55],   # of I2,5
	 [0.40, 0.60],   # of I2,6
	 [0.00, 0.00],   # of I2,7 ... terminal nodes <- mock-up zero strategy
	 [0.00, 0.00]],  # of I2,8 ... imaginary nodes <- mock-up zero strategy
	name="current_infoset_strategies_lvl2"
)

########## Level 3 ##########
# There are never any infosets in the final layer, only terminal / imaginary nodes.
shape[3] = actions_per_levels[:3]
node_types_lvl3_tmp = tf.Variable(
	tf.fill(
		value=TERMINAL_NODE,
		dims=shape[3]),
	name="node_types_lvl3"
)
indices_imaginary_nodes_lvl3_tmp = tf.constant(
	[[0, 2],   # children of s7
	 [1, 2],   # children of s10
	 [2, 2],   # children of s13
	 [3, 2],   # children of s16
	 [4, 0]],  # children of s17
	name="indices_imaginary_nodes_lvl3"
)
node_types_lvl3_tmp = tf.scatter_nd_update(
	ref=node_types_lvl3_tmp,
	indices=indices_imaginary_nodes_lvl3_tmp,
	updates=tf.fill(
		value=IMAGINARY_NODE,
		dims=indices_imaginary_nodes_lvl3_tmp.shape
	),
	name="node_types_lvl3"
)
node_types[3] = node_types_lvl3_tmp
utilities_lvl3_tmp = tf.Variable(
	tf.fill(
		value=NON_TERMINAL_UTILITY,
		dims=shape[3]
	)
)
mask_terminals_lvl3_tmp = tf.equal(node_types[3], TERMINAL_NODE)
terminal_values_lvl3_tmp = tf.reshape(tf.range(10, 310, delta=10.0), shape[3])
utilities_lvl3_tmp = masked_assign(
	ref=utilities_lvl3_tmp,
	value=terminal_values_lvl3_tmp,
	mask=mask_terminals_lvl3_tmp,
	name="utilities_lvl3"
)
utilities[3] = utilities_lvl3_tmp


########## miscellaneous tensors ##########
cf_values_infoset_actions = [
	# TODO rewrite to list comprehension
	tf.Variable(tf.zeros_like(current_infoset_strategies[0]), name="cf_values_infoset_actions_lvl0"),
	tf.Variable(tf.zeros_like(current_infoset_strategies[1]), name="cf_values_infoset_actions_lvl1"),
	tf.Variable(tf.zeros_like(current_infoset_strategies[2]), name="cf_values_infoset_actions_lvl2")
]
positive_cumulative_regrets = [
	# TODO rewrite to list comprehension
	tf.Variable(tf.zeros_like(current_infoset_strategies[0]), name="positive_cumulative_regrets_lvl0"),
	tf.Variable(tf.zeros_like(current_infoset_strategies[1]), name="positive_cumulative_regrets_lvl1"),
	tf.Variable(tf.zeros_like(current_infoset_strategies[2]), name="positive_cumulative_regrets_lvl2")
]
cumulative_infoset_strategies = [tf.Variable(tf.zeros_like(current_infoset_strategies[level]),
                                             name="cumulative_infoset_strategies_lvl{}".format(level))
                                 for level in range(acting_depth)]         # used for the final average strategy
infosets_of_non_chance_player = [
	tf.reshape(tf.not_equal(infoset_acting_players[level], CHANCE_PLAYER),
	           shape=[current_infoset_strategies[level].shape[0]],
	           name="infosets_of_acting_player_lvl{}".format(level))
	for level in range(acting_depth)
]
cfr_step = tf.Variable(initial_value=0, dtype=tf.int64, name="cfr_step")   # counter of CFR+ iterations
averaging_delay = tf.constant(         # https://arxiv.org/pdf/1407.5042.pdf (Figure 2)
		0,  # averaging delay of 0 means that there is no delay, or use DEFAULT_AVERAGING_DELAY,
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
		y=-1.0,                   # Opponent's utilities in zero-sum games = (-utilities) of `player_owning_the_utilities`
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
