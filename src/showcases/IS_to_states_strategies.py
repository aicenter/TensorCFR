#!/usr/bin/env python3

import tensorflow as tf

from assign_strategies_to_states import assign_strategies_to_states
from domain.domain_01 import IS_strategies_lvl0, state2IS_lvl0, IS_strategies_lvl1, state2IS_lvl1, IS_strategies_lvl2, \
	state2IS_lvl2
from utils.tensor_utils import print_tensors

# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

########## Level 0 ##########
# I0,0 = {} ... special index - all-1's strategy for counterfactual probability
# I0,1 = { s } ... root state, the opponent acts here
state_strategies_lvl0 = assign_strategies_to_states(IS_strategies_lvl0, state2IS_lvl0, name="state_strategies_lvl0")

########## Level 1 ##########
# I1,0 = { s' } ... special index - all-1's strategy for counterfactual probability
# I1,1 = { s1 }
# I1,2 = { s2, s3 }
# I1,3 = Ic = { s4 } ... chance node
state_strategies_lvl1 = assign_strategies_to_states(IS_strategies_lvl1, state2IS_lvl1, name="state_strategies_lvl1")

########## Level 2 ##########
# I2,0 = { s5, s8, s9, s18 } ... special index - all-1's strategy for counterfactual probability
# I2,1 = { s6  }
# I2,2 = { s11, s14 }
# I2,3 = { s12, s15 } ... chance nodes
# I2,4 = { s19 }
# I2,t = { s7, s10, s13, s16, s17 } ... terminal nodes
state_strategies_lvl2 = assign_strategies_to_states(IS_strategies_lvl2, state2IS_lvl2, name="state_strategies_lvl2")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# more verbose output
	print("########## Level 0 ##########")
	print_tensors(sess, [state2IS_lvl0, IS_strategies_lvl0, state_strategies_lvl0])
	print("########## Level 1 ##########")
	print_tensors(sess, [state2IS_lvl1, IS_strategies_lvl1, state_strategies_lvl1])
	print("########## Level 2 ##########")
	print_tensors(sess, [state2IS_lvl2, IS_strategies_lvl2, state_strategies_lvl2])
