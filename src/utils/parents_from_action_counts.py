#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


from src.commons.constants import INT_DTYPE
from pprint import pprint
from src.utils.tensor_utils import print_tensors, print_tensor, scatter_nd_sum


def get_parents_from_action_counts_alternative(action_counts):
	levels = len(action_counts)
	max_actions = list(map(np.amax, action_counts))
	print("levels:")
	pprint(levels, indent=1, width=80)
	print("max_actions:")
	pprint(max_actions, indent=1, width=80)

	mask_children = [
		tf.sequence_mask(
				action_counts[level],
				name="mask_children_lvl{}".format(level)
		)
		for level in range(levels)
	]
	expanded_ranges = [
		tf.range(
				start=0,
				limit=len(action_counts[level]),
				dtype=INT_DTYPE,
				name="expanded_range_lvl{}".format(level),
		)
		for level in range(levels)
	]
	with tf.Session() as tmp_sess:
		tmp_sess.run(tf.global_variables_initializer())
		print_tensors(tmp_sess, mask_children)
		print_tensors(tmp_sess, expanded_ranges)


def get_parents_from_action_counts(action_counts):
	"""
	Compute tensor `parents` containing the indices to each node's parent in the previous level.

  Args:
    :param action_counts: A 2-D numpy array containing number of actions of each node.

  Returns:
    A corresponding TensorFlow operation (from the computation graph) that contain the index to each node's parent
     (in the level above).
  """

	# the 1st size is `1` as there's only 1 root, the last `action_count` is skipped as there are only terminal nodes
	sizes = [1] + list(map(np.sum, action_counts[:-1]))
	print("sizes:")
	pprint(sizes, indent=1, width=50)

	leftmost_child = [    # indices of each node's leftmost child
		tf.cast(
				tf.cumsum(
						action_counts[level],
						exclusive=False,
						name="leftmost_child_lvl{}".format(level)
				),
				dtype=INT_DTYPE,
		)
		for level in range(len(action_counts))
	]

	parents = [
		tf.sparse_to_dense(
				sparse_indices=tf.cast(leftmost_child[level - 1] if level > 0 else [], dtype=INT_DTYPE),
				output_shape=[sizes[level]],
				sparse_values=1,
				default_value=0,
				name="parents_lvl{}".format(level),
		)
		for level in range(len(sizes))
	]

	with tf.Session() as tmp_sess:
		tmp_sess.run(tf.global_variables_initializer())
		print_tensors(tmp_sess, leftmost_child)

	return parents


if __name__ == '__main__':
	""" Demonstrate on `domains.hunger_games`:
	
	TODO
	"""

	action_counts_ = [
		[2],
		[1, 6],
		[4, 0, 0, 0, 0, 0, 0],
		[3, 3, 2, 2],
		[2] * 10,
		[0] * 20
	]
	print("action_counts:")
	pprint(action_counts_, indent=1, width=80)

	get_parents_from_action_counts_alternative(action_counts_)

	# parents_ = get_parents_from_action_counts(action_counts_)
	#
	# with tf.Session() as sess:
	# 	sess.run(tf.global_variables_initializer())
	# 	print_tensors(sess, parents_)
