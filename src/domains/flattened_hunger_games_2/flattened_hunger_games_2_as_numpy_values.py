#!/usr/bin/env python3
from typing import List

from src.commons.constants import NON_TERMINAL_UTILITY, PLAYER1, PLAYER2, \
	IMAGINARY_PROBABILITIES, INFOSET_FOR_TERMINAL_NODES

# flattened domain `hunger_games`: see `doc/hunger_games_via_drawing.png` and `doc/hunger_games_2/`

action_counts = [
	[2],
	[1, 6],
	[4, 0, 0, 0, 0, 0, 0],
	[3, 3, 2, 2],
	[2] * 10,
	[0] * 20,
]
levels = len(action_counts)
acting_depth = len(action_counts) - 1  # the last level has only (non-acting) terminal nodes

# allocate Python arrays
node_to_infoset: List[int] = [None] * acting_depth
utilities: List[int] = [None] * levels
infoset_acting_players: List[int] = [None] * acting_depth
initial_infoset_strategies: List[int] = [None] * acting_depth

########## Level 0 ##########
node_to_infoset[0] = [0]
utilities[0] = [NON_TERMINAL_UTILITY]   # TODO to make bottom-up work but this needs to be discussed
infoset_acting_players[0] = [PLAYER1]
initial_infoset_strategies[0] = [[0.1, 0.9]]

########## Level 1 ##########
node_to_infoset[1] = [0, 1]
utilities[1] = [NON_TERMINAL_UTILITY] * 2
infoset_acting_players[1] = [PLAYER2, PLAYER2]
initial_infoset_strategies[1] = [
	[1.0, IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES,
	 IMAGINARY_PROBABILITIES],  # TODO assign infoset strategies by action counts
	[0.1, 0.1, 0.1, 0.0, 0.2, 0.5]
]

########## Level 2 ##########
node_to_infoset[2] = [0, INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES,
                      INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES, INFOSET_FOR_TERMINAL_NODES]
utilities[2] = [NON_TERMINAL_UTILITY, -6, -8, -5, -10, -3, -1]
infoset_acting_players[2] = [PLAYER1]
initial_infoset_strategies[2] = [
	[0.1, 0.2, 0.0, 0.7]
]

########## Level 3 ##########
node_to_infoset[3] = [0, 0, 1, 1]
utilities[3] = [NON_TERMINAL_UTILITY] * 4
infoset_acting_players[3] = [PLAYER2, PLAYER2]
initial_infoset_strategies[3] = [
	[0.1, 0.0, 0.9],
	[0.2, 0.8, IMAGINARY_PROBABILITIES]  # TODO assign infoset strategies by action counts
]

########## Level 4 ##########
node_to_infoset[4] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
utilities[4] = [NON_TERMINAL_UTILITY] * 10
infoset_acting_players[4] = [
	PLAYER1,
	PLAYER1,
	PLAYER1,
	PLAYER1,
	PLAYER1,
	PLAYER1,
	PLAYER1,
	PLAYER1,
	PLAYER1,
	PLAYER1
]
initial_infoset_strategies[4] = [
	[0.1, 0.9],
	[0.2, 0.8],
	[0.3, 0.7],
	[0.4, 0.6],
	[0.5, 0.5],
	[0.6, 0.4],
	[0.7, 0.3],
	[0.8, 0.2],
	[0.9, 0.1],
	[1.0, 0.0]
]

########## Level 5 ##########
# There are never any infosets in the final layer, only terminal / imaginary nodes.
utilities[5] = [86, 5, 31, 30, 1, 1, 10, 10, 99, 100, 45, 4, 16, 25, 77, 80, 78, 0, 9, 1]

if __name__ == '__main__':
	from pprint import pprint
	print("action_counts:")
	pprint(action_counts, indent=1, width=80)
	print("levels:")
	pprint(levels, indent=1, width=80)
	print("acting_depth:")
	pprint(acting_depth, indent=1, width=80)
	print("node_to_infoset:")
	pprint(node_to_infoset, indent=1, width=80)
	print("utilities:")
	pprint(utilities, indent=1, width=180)
	print("infoset_acting_players:")
	pprint(infoset_acting_players, indent=1, width=40)
	print("initial_infoset_strategies:")
	pprint(initial_infoset_strategies, indent=1, width=35)
