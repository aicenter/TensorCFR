#!/usr/bin/env python3

# utilities
import numpy as np
import tensorflow as tf

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

# test-error tolerances
LARGE_ERROR_TOLERANCE = 0.0001
SMALL_ERROR_TOLERANCE = 0.0000001

# values for default settings
DEFAULT_AVERAGING_DELAY = 250
DEFAULT_TOTAL_STEPS = 1000
DEFAULT_TOTAL_STEPS_ON_SMALL_DOMAINS = 50

# Gambit Node Types
GAMBIT_NODE_TYPE_TERMINAL = 't'
GAMBIT_NODE_TYPE_CHANCE = 'c'
GAMBIT_NODE_TYPE_PLAYER = 'p'

INT_DTYPE = tf.int32
IMAGINARY_PROBABILITIES = 0.0
