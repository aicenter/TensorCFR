#!/usr/bin/env python3

import tensorflow as tf

from assign_strategies_to_nodes import assign_strategies_to_nodes
from domains.domain01.domain_01 import IS_strategies, node_to_IS
from utils.tensor_utils import print_tensors

# custom-made game: doc/domain_01_via_drawing.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)
# TODO extract to a test

node_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies[0], node_to_IS[0], name="node_strategies_lvl0")
node_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies[1], node_to_IS[1], name="node_strategies_lvl1")
node_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies[2], node_to_IS[2], name="node_strategies_lvl2")

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    # more verbose output
    print("########## Level 0 ##########")
    print_tensors(sess, [node_to_IS[0], IS_strategies[0], node_strategies_lvl0])
    print("########## Level 1 ##########")
    print_tensors(sess, [node_to_IS[1], IS_strategies[1], node_strategies_lvl1])
    print("########## Level 2 ##########")
    print_tensors(sess, [node_to_IS[2], IS_strategies[2], node_strategies_lvl2])
