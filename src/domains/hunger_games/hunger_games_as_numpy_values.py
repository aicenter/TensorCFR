#!/usr/bin/env python3
from typing import List
from pprint import pprint

import numpy as np

from src.commons.constants import NON_TERMINAL_UTILITY, INNER_NODE, TERMINAL_NODE, PLAYER1, PLAYER2, \
	IMAGINARY_PROBABILITIES, IMAGINARY_NODE, NO_ACTING_PLAYER

# domain `hunger_games`: see doc/hunger_games_via_drawing.png

actions_per_levels = [2, 6, 4, 3, 2]
levels = len(actions_per_levels) + 1  # accounting for 0th level
acting_depth = len(actions_per_levels)

# allocate Python arrays
node_to_infoset: List[int] = [None] * acting_depth
node_types: List[int] = [None] * levels
utilities: List[int] = [None] * levels
infoset_acting_players: List[int] = [None] * acting_depth
initial_infoset_strategies: List[int] = [None] * acting_depth

########## Level 0 ##########
node_to_infoset[0] = 0
node_types[0] = INNER_NODE
utilities[0] = NON_TERMINAL_UTILITY
infoset_acting_players[0] = [PLAYER1]
initial_infoset_strategies[0] = [[0.1, 0.9]]

########## Level 1 ##########
node_to_infoset[1] = [0, 1]
node_types[1] = [INNER_NODE] * 2
utilities[1] = [NON_TERMINAL_UTILITY] * 2
infoset_acting_players[1] = [PLAYER2, PLAYER2]
initial_infoset_strategies[1] = [
	[1.0, IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES, IMAGINARY_PROBABILITIES,
	 IMAGINARY_PROBABILITIES],
	[0.1, 0.1, 0.1, 0.0, 0.2, 0.5]
]

########## Level 2 ##########
node_to_infoset[2] = [
	[0, 2, 2, 2, 2, 2],   # `2` for the infoset of imaginary nodes
	[1, 1, 1, 1, 1, 1]    # `1` for the infoset of terminal nodes
]
node_types[2] = [
	[INNER_NODE, IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
	[TERMINAL_NODE, TERMINAL_NODE, TERMINAL_NODE, TERMINAL_NODE, TERMINAL_NODE, TERMINAL_NODE]
]
utilities[2] = [
	[NON_TERMINAL_UTILITY] * 6,
	[-1.0] * 6
]
infoset_acting_players[2] = [PLAYER1, NO_ACTING_PLAYER, NO_ACTING_PLAYER]
initial_infoset_strategies[2] = [
	[0.1, 0.2, 0.0, 0.7],
	[np.nan] * 4,
	[np.nan] * 4
]

########## Level 3 ##########
node_to_infoset[3] = [
	[
		[0, 0, 1, 1],
		[2] * 4,   # `2` for the infoset of imaginary nodes
		[2] * 4,
		[2] * 4,
		[2] * 4,
		[2] * 4
	],
	[
		[2] * 4,
		[2] * 4,
		[2] * 4,
		[2] * 4,
		[2] * 4,
		[2] * 4
	]
]
node_types[3] = [
	[
		[TERMINAL_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4
	],
	[
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4,
		[IMAGINARY_NODE] * 4
	]
]
utilities[3] = [
	[
		[1.0] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
	],
	[
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4,
		[NON_TERMINAL_UTILITY] * 4
	]
]
infoset_acting_players[3] = [PLAYER2, PLAYER2, NO_ACTING_PLAYER]
initial_infoset_strategies[3] = [
	[0.1, 0.0, 0.9],
	[np.nan] * 3,
	[np.nan] * 3
]

########## Level 4 ##########
# There are never any infosets in the final layer, only terminal / imaginary nodes.
node_to_infoset[4] = [
	[
		[
			[0, 1,  2],
			[3, 4,  5],
			[6, 7, 10],
			[8, 9, 10]
		],
		[
			[10] * 3,   # `10` for the infoset of imaginary nodes
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		]
	],
	[
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
		[
			[10] * 3,
			[10] * 3,
			[10] * 3,
			[10] * 3,
		],
	]
]
node_types[4] = [
	[
		[
			[INNER_NODE, INNER_NODE,     INNER_NODE],
			[INNER_NODE, INNER_NODE,     INNER_NODE],
			[INNER_NODE, INNER_NODE, IMAGINARY_NODE],
			[INNER_NODE, INNER_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		]
	],
	[
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		],
		[
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
			[IMAGINARY_NODE, IMAGINARY_NODE, IMAGINARY_NODE],
		]
	]
]
utilities[4] = [
	[
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		]
	],
	[
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		],
		[
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
			[NON_TERMINAL_UTILITY] * 3,
		]
	]
]


# TODO continue here
pprint(utilities[4], indent=1, width=30)
