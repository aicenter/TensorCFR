#!/usr/bin/env python3

import tensorflow as tf

from src.domains.domain01.domain01 import immediate_infoset_strategies, node_to_infoset, acting_depth
from src.domains.domain01.node_strategies import assign_strategies_to_nodes
from src.utils.tensor_utils import print_tensors

# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png
# TODO extract to a test

# node_strategies = [None] * acting_depth
node_strategies = [
	assign_strategies_to_nodes(
			immediate_infoset_strategies[level],
			node_to_infoset[level],
			name="node_strategies_lvl{}".format(level)
	) for level in range(acting_depth)
]

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	# more verbose output
	for level in range(acting_depth):
		print("########## Level {} ##########".format(level))
		print_tensors(sess, [node_to_infoset[level], immediate_infoset_strategies[level], node_strategies[level]])
