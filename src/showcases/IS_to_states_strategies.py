#!/usr/bin/env python3

import tensorflow as tf

from src.domains.domain01.node_strategies import assign_strategies_to_nodes
from src.domains.domain01.domain_01 import IS_strategies_lvl0, node_to_IS_lvl0, IS_strategies_lvl1, node_to_IS_lvl1, \
	IS_strategies_lvl2, \
	node_to_IS_lvl2
from src.utils.tensor_utils import print_tensors

# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png
# TODO extract to a test

node_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies_lvl0, node_to_IS_lvl0, name="node_strategies_lvl0")
node_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies_lvl1, node_to_IS_lvl1, name="node_strategies_lvl1")
node_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies_lvl2, node_to_IS_lvl2, name="node_strategies_lvl2")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# more verbose output
	print("########## Level 0 ##########")
	print_tensors(sess, [node_to_IS_lvl0, IS_strategies_lvl0, node_strategies_lvl0])
	print("########## Level 1 ##########")
	print_tensors(sess, [node_to_IS_lvl1, IS_strategies_lvl1, node_strategies_lvl1])
	print("########## Level 2 ##########")
	print_tensors(sess, [node_to_IS_lvl2, IS_strategies_lvl2, node_strategies_lvl2])
