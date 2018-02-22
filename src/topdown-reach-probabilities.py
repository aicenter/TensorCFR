#!/usr/bin/env python3

import tensorflow as tf

from domain.domain_01 import reach_probabilities_lvl0, state2IS_lvl0, IS_strategies_lvl0, state2IS_lvl1, \
	IS_strategies_lvl1, state2IS_lvl2, IS_strategies_lvl2
from utils.tensor_utils import print_tensors, expanded_multiply

# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

# tensors to be computed at level 0
state_strategies_lvl0 = tf.gather(params=IS_strategies_lvl0, indices=state2IS_lvl0, name="state_strategies_lvl0")

reach_probabilities_lvl1 = expanded_multiply(expandable_tensor=reach_probabilities_lvl0,
                                             expanded_tensor=state_strategies_lvl0, name="reach_probabilities_lvl1")

# tensors to be computed at level 1
state_strategies_lvl1 = tf.gather(params=IS_strategies_lvl1, indices=state2IS_lvl1, name="state_strategies_lvl1")
reach_probabilities_lvl2 = expanded_multiply(expandable_tensor=reach_probabilities_lvl1,
                                             expanded_tensor=state_strategies_lvl1, name="reach_probabilities_lvl2")

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
