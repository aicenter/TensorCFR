#!/usr/bin/env python3

from src.commons.constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, IMAGINARY_NODE, CHANCE_PLAYER, \
	PLAYER1, PLAYER2, NO_ACTING_PLAYER, IMAGINARY_PROBABILITIES

# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

actions_per_levels = [5, 3, 2]  # maximum number of actions per each level (0, 1, 2)
levels = len(actions_per_levels) + 1  # accounting for 0th level
acting_depth = len(actions_per_levels)

# Init
node_to_infoset = [None] * acting_depth
node_types = [None] * levels
utilities = [None] * levels
infoset_acting_players = [None] * acting_depth
initial_infoset_strategies = [None] * acting_depth

########## Level 0 ##########
# I0,0 = { s } ... root node, the chance player acts here
# there are 5 actions in node s
node_to_infoset[0] = 0
node_types[0] = INNER_NODE
utilities[0] = NON_TERMINAL_UTILITY
infoset_acting_players[0] = [CHANCE_PLAYER]
initial_infoset_strategies[0] = [[0.5, .25, 0.1, 0.1, .05]]  # of I0,0

########## Level 1 ##########
# I1,0 = { s' }
# I1,1 = { s1 }
# I1,2 = { s2, s3 }
# I1,3 = { s4 } ... chance node
# each node has 3 actions
node_to_infoset[1] = [0, 1, 2, 2, 3]
node_types[1] = [INNER_NODE] * 5
utilities[1] = [NON_TERMINAL_UTILITY] * 5
infoset_acting_players[1] = [
	PLAYER1,         # I1,0
	PLAYER2,         # I1,1
	PLAYER2,         # I1,2
	CHANCE_PLAYER    # I1,3
]
initial_infoset_strategies[1] = [
	[0.5, 0.4, 0.1],                      # of I1,0
	[0.1, 0.9, IMAGINARY_PROBABILITIES],  # of I1,1, `nan` for probabilities of imaginary nodes
	[0.2, 0.8, IMAGINARY_PROBABILITIES],  # of I1,2, `nan` for probabilities of imaginary nodes
	[0.3, 0.3, 0.3]                       # of I1,3
]

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
node_to_infoset[2] = [
											 [0, 1, 7],   # s5, s6, s7
											 [2, 2, 8],   # s8, s9, s10
											 [3, 4, 8],   # s11, s12, s1
											 [3, 4, 8],   # s14, s15, s16
											 [7, 5, 6]    # s17, s18, s19
										 ]
node_types[2] = [
									[INNER_NODE, INNER_NODE, TERMINAL_NODE],   # s5, s6, s7
									[INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s8, s9, s10
									[INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s11, s12, s13
									[INNER_NODE, INNER_NODE, IMAGINARY_NODE],  # s14, s15, s16
									[TERMINAL_NODE, INNER_NODE, INNER_NODE]    # s17, s18, s19
								]
utilities[2] = [
	[  0.,   0.,  30.],
	[  0.,   0.,   0.],
	[  0.,   0.,   0.],
	[  0.,   0.,   0.],
	[130.,   0.,   0.]
]
infoset_acting_players[2] = [
	PLAYER1,            # of I2,0
	PLAYER2,            # of I2,1
	PLAYER1,            # of I2,2
	PLAYER2,            # of I2,3
	CHANCE_PLAYER,      # of I2,4
	PLAYER1,            # of I2,5
	PLAYER2,            # of I2,6
	NO_ACTING_PLAYER,   # of I2,7 ... pseudo-infoset of terminal nodes
	NO_ACTING_PLAYER    # of I2,8 ... pseudo-infoset of imaginary nodes
]
initial_infoset_strategies[2] = [
	[0.15, 0.85],  # of I2,0
	[0.70, 0.30],  # of I2,1
	[0.25, 0.75],  # of I2,2
	[0.50, 0.50],  # of I2,3
	[0.10, 0.90],  # of I2,4
	[0.45, 0.55],  # of I2,5
	[0.40, 0.60],  # of I2,6
	[IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES],   # of I2,7 ... terminal nodes
	[IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES]    # of I2,8 ... imaginary nodes
]

########## Level 3 ##########
# There are never any infosets in the final layer, only terminal / imaginary nodes.
node_types[3] = [
	[[1, 1],
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
	 [1, 1]]
]
utilities[3] = [
	[
		[ 10.,  20.],
		[ 30.,  40.],
		[  0.,   0.]
	],

	[
		[ 70.,  80.],
		[ 90., 100.],
		[  0.,   0.]
	],

	[
		[130., 140.],
		[150., 160.],
		[  0.,   0.]
	],

	[
		[190., 200.],
		[210., 220.],
		[  0.,   0.]
	],

	[
		[  0.,   0.],
		[270., 280.],
		[290., 300.]
	]
]
