#!/usr/bin/env python3

import tensorflow as tf

from utils.tensor_utils import print_tensors

# 3 actions, 1 layer looks like:
#   I0 ... special index - all-1's strategy for counterfactual probability
#   I1 = { s1 }
#   I2 = { s2, s3 }
#   I3 = Ic = { s4 } -- chance node
state2IS = tf.Variable([0, 1, 2, 2, 3], name="state2IS")
IS_strategies = tf.Variable([[1.0, 1.0, 1.0],   # of I0
                             [0.1, 0.9, 0.0],   # of I1
                             [0.2, 0.8, 0.0],   # of I2
                             [0.3, 0.3, 0.3]],  # of Ic
                            name="IS_strategies")
state_strategies = tf.gather(params=IS_strategies, indices=state2IS, name="state_strategies")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [state2IS, IS_strategies, state_strategies])
