#!/usr/bin/env python3

# utilities
NON_TERMINAL_UTILITY = 0.0

# type of nodes
INNER_NODE = 0
TERMINAL_NODE = 1
IMAGINARY_NODE = 2

# type of players
PLAYER1 = 1
PLAYER2 = 2
CHANCE_PLAYER = 0
NO_ACTING_PLAYER = -1  # dummy acting-player value in nodes without acting players, i.e. terminal and imaginary nodes

# Test error tolerances
LARGE_ERROR_TOLERANCE = 0.0001
SMALL_ERROR_TOLERANCE = 0.0000001
DEFAULT_AVERAGING_DELAY = 250
