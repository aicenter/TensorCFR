#!/usr/bin/env python3

from src.commons.constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, PLAYER1, PLAYER2

# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

actions_per_levels = [2, 2]
levels = len(actions_per_levels) + 1  # accounting for 0th level
acting_depth = len(actions_per_levels)

# Init
node_to_infoset = [None] * acting_depth
node_types = [None] * levels
utilities = [None] * levels
infoset_acting_players = [None] * acting_depth
initial_infoset_strategies = [None] * acting_depth

########## Level 0 ##########
node_to_infoset[0] = 0
node_types[0] = INNER_NODE
utilities[0] = NON_TERMINAL_UTILITY
infoset_acting_players[0] = [PLAYER1]
initial_infoset_strategies[0] = [[0.1, 0.9]]

########## Level 1 ##########
node_to_infoset[1] = [0, 0]
node_types[1] = [INNER_NODE] * 2
utilities[1] = [NON_TERMINAL_UTILITY] * 2
infoset_acting_players[1] = [PLAYER2]
initial_infoset_strategies[1] = [[0.2, 0.8]]

########## Level 2 ##########
# There are never any infosets in the final layer, only terminal / imaginary nodes.
node_types[2] = [
	[TERMINAL_NODE, TERMINAL_NODE],
	[TERMINAL_NODE, TERMINAL_NODE]
]
utilities[2] = [
	[-1.0, 1.0],
	[1.0, -1.0]
]
