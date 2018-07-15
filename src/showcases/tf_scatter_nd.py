#!/usr/bin/env python3

import tensorflow as tf

from src.domains.flattened_hunger_games.flattened_hunger_games_as_numpy_values import infoset_acting_players
from src.utils.tensor_utils import print_tensors

action_counts = [
	[2],
	[1, 6],
	[4, 0, 0, 0, 0, 0, 0],
	[3, 3, 2, 2],
	[2] * 10,
	[0] * 20,
]
node_to_infoset = [
	[0],
	[0, 1],
	[0, 1, 1, 1, 1, 1, 1],    # `1` for the infoset of terminal nodes
	[0, 0, 1, 1],
	[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
]
mask_of_inner_nodes = [
	tf.not_equal(
		action_count,
		0,
		name="mask_of_inner_nodes_lvl{}".format(level)
	)
	for level, action_count in enumerate(action_counts)
]
inner_node_to_infoset = [
	tf.expand_dims(
		tf.boolean_mask(
			node_to_infoset_level,
			mask=mask_of_inner_nodes[level]
		),
		axis=-1,
		name="inner_node_to_infoset_lvl{}".format(level),
	)
	for level, node_to_infoset_level in enumerate(node_to_infoset)
]
# infoset_action_counts = [
# 	tf.scatter_nd(
# 		indices=inner_node_to_infoset[level],
# 		updates=tf.constant(action_counts[level]),
# 		shape=tf.shape(infoset_acting_players[level]),
# 		name="infoset_action_counts_lvl{}".format(level),
# 	)
# 	for level in range(len(infoset_acting_players))
# ]


if __name__ == '__main__':
	# print("action_counts:")
	# pprint(action_counts, indent=1, width=80)
	# print("node_to_infoset:")
	# pprint(node_to_infoset, indent=1, width=80)
	# print("infoset_acting_players:")
	# pprint(infoset_acting_players, indent=1, width=40)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())

		for level in range(len(infoset_acting_players)):
			print("########## Level {} ##########".format(level))
			print_tensors(sess, [
				tf.expand_dims(
					node_to_infoset[level],
					axis=-1,
					name="expanded_node_to_infoset_indices_lvl{}".format(level)
				),
				tf.constant(
					action_counts[level],
					name="action_counts_lvl{}".format(level)
				),
				tf.shape(
					infoset_acting_players[level],
					name="infoset_acting_players_lvl{}".format(level)
				),
				mask_of_inner_nodes[level]
			])
		# 	print_tensors(sess, [infoset_action_counts[level]])
		print("##############################")
		print_tensors(sess, mask_of_inner_nodes)
		print("______________________________")
		print_tensors(sess, inner_node_to_infoset)
		# print_tensors(sess, infoset_action_counts)
