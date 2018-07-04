import tensorflow as tf
import numpy as np


TERMINAL_NODE = 't'
CHANCE_NODE = 'c'
PLAYER_NODE = 'p'

INT_DTYPE_NUMPY = np.int32
INT_DTYPE = tf.as_dtype(INT_DTYPE_NUMPY)
FLOAT_DTYPE = tf.float32
IMAGINARY_PROBABILITIES = 0.0