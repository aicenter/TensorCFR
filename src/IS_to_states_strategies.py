#!/usr/bin/env python3

import tensorflow as tf

# I0 ... special index - all-1's strategy for counterfactual probability
# I1 = { s1 }
# I2 = { s2, s3 }
# Ic = { s4 } -- chance node
from utils.tensor_utils import print_tensor

state2IS = tf.Variable([0, 1, 2, 2, 4], name="state2IS")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensor(sess, state2IS)
