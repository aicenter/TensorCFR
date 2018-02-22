#!/usr/bin/env python3

import tensorflow as tf

from utils.tensor_utils import print_tensors, expanded_multiply

# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

########## Level 0 ##########
# I0,0 = {} ... special index - all-1's strategy for counterfactual probability
# I0,1 = { s } ... root state, the opponent acts here
# there are 5 actions in state s
reach_probabilities_lvl0 = tf.Variable(1.0, name="reach_probabilities_lvl0")
state2IS_lvl0 = tf.Variable(1, name="state2IS_lvl0")
# NOTE: the value above is not [1] in order to remove 1 redundant '[]' represented by choice of empty sequence {}
IS_strategies_lvl0 = tf.Variable([[1.0, 1.0, 1.0, 1.0, 1.0],   # of I0,0
                                  [0.5, .25, 0.1, 0.1, .05]],  # of I0,1
                                 name="IS_strategies_lvl0")
# tensors to be computed at level 0
state_strategies_lvl0 = tf.gather(params=IS_strategies_lvl0, indices=state2IS_lvl0, name="state_strategies_lvl0")

reach_probabilities_lvl1 = expanded_multiply(expandable_tensor=reach_probabilities_lvl0,
                                             expanded_tensor=state_strategies_lvl0, name="reach_probabilities_lvl1")

########## Level 1 ##########
# I1,0 = { s' } ... special index - all-1's strategy for counterfactual probability
# I1,1 = { s1 }
# I1,2 = { s2, s3 }
# I1,3 = Ic = { s4 } ... chance node
# each state 3 actions
state2IS_lvl1 = tf.Variable([0, 1, 2, 2, 3], name="state2IS_lvl1")
IS_strategies_lvl1 = tf.Variable([[1.0, 1.0, 1.0],   # of I1,0
                                  [0.1, 0.9, 0.0],   # of I1,1
                                  [0.2, 0.8, 0.0],   # of I1,2
                                  [0.3, 0.3, 0.3]],  # of I1,c
                                 name="IS_strategies_lvl1")
# tensors to be computed at level 1
state_strategies_lvl1 = tf.gather(params=IS_strategies_lvl1, indices=state2IS_lvl1, name="state_strategies_lvl1")
reach_probabilities_lvl2 = expanded_multiply(expandable_tensor=reach_probabilities_lvl1,
                                             expanded_tensor=state_strategies_lvl1, name="reach_probabilities_lvl2")

########## Level 2 ##########
# I2,0 = { s5, s8, s9, s18 } ... special index - all-1's strategy for counterfactual probability
# I2,1 = { s6  }
# I2,2 = { s11, s14 }
# I2,3 = { s12, s15 } ... chance nodes
# I2,4 = { s19 }
# I2,t = { s7, s10, s13, s16, s17 } ... terminal nodes
# each state 2 actions
state2IS_lvl2 = tf.Variable([[0, 1, 5],   # s5, s6, s7
                             [0, 0, 5],   # s8, s9, s10
                             [2, 3, 5],   # s11, s12, s13
                             [2, 3, 5],   # s14, s15, s16
                             [5, 0, 4]],  # s17, s18, s19
                            name="state2IS_lvl2")
IS_strategies_lvl2 = tf.Variable([[1.0, 1.0],   # of I2,0
                                  [0.7, 0.3],   # of I2,1
                                  [0.5, 0.5],   # of I2,2
                                  [0.1, 0.9],   # of I2,3 ... chance player
                                  [0.4, 0.6],   # of I2,4
                                  [0.0, 0.0]],  # of I2,t ... no strategies terminal nodes <- mock-up strategy
                                 name="IS_strategies_lvl2")
# tensors to be computed at level 2
state_strategies_lvl2 = tf.gather(params=IS_strategies_lvl2, indices=state2IS_lvl2, name="state_strategies_lvl2")
reach_probabilities_lvl3 = expanded_multiply(expandable_tensor=reach_probabilities_lvl2,
                                             expanded_tensor=state_strategies_lvl2, name="reach_probabilities_lvl3")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# more verbose output
	print("########## Level 0 ##########")
	print_tensors(sess, [reach_probabilities_lvl0, state2IS_lvl0, IS_strategies_lvl0, state_strategies_lvl0])
	print("########## Level 1 ##########")
	print_tensors(sess, [reach_probabilities_lvl1, state2IS_lvl1, IS_strategies_lvl1, state_strategies_lvl1])
	print("########## Level 2 ##########")
	print_tensors(sess, [reach_probabilities_lvl2, state2IS_lvl2, IS_strategies_lvl2, state_strategies_lvl2])
	print("########## Level 3 ##########")
	print_tensors(sess, [reach_probabilities_lvl3])
