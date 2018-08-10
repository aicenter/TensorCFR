#!/usr/bin/env python3
import os

import numpy as np
import tensorflow as tf

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

INT_DTYPE_NUMPY = np.int32
INT_DTYPE = tf.as_dtype(INT_DTYPE_NUMPY)
FLOAT_DTYPE = tf.float32

# constants for probabilities
IMAGINARY_PROBABILITIES = 0.0
REACH_PROBABILITY_OF_ROOT = 1.0

# constants for indices
INFOSET_FOR_TERMINAL_NODES = -1

# absolute path to the root of the project source code
PROJECT_ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

RANDOM_SEED = 1337
DEFAULT_DATASET_SIZE = 3
