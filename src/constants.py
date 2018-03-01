#!/usr/bin/env python3

# utilities
NON_TERMINAL_UTILITY = 0.0

# type of nodes
INNER_NODE = 0
TERMINAL_NODE = 1
IMAGINARY_NODE = 2

# type of players
UPDATING_PLAYER = 1
OPPONENT = 2
CHANCE_PLAYER = 0
NO_ACTING_PLAYER = -1  # dummy acting-player value in nodes without acting players, i.e. terminal and imaginary nodes
