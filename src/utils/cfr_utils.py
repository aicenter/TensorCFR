#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


from src.commons.constants import INT_DTYPE
from pprint import pprint
from src.utils.tensor_utils import print_tensors, print_tensor, scatter_nd_sum


def get_parents_from_action_counts(action_counts):
	"""
	Compute tensor `parents` containing the indices to each node's parent in the previous level.

  Args:
    :param action_counts: A 2-D numpy array containing number of actions of each node.

  Returns:
    A corresponding TensorFlow operation (from the computation graph) that contain the index to each node's parent
     (in the level above).
  """
	levels = len(action_counts)
	max_actions = list(map(np.amax, action_counts))
	mask_children = [
		tf.sequence_mask(
				action_counts[level - 1],
				name="mask_children_lvl{}".format(level)
		)
		for level in range(1, levels)
	]
	broadcast_ranges = [
		tf.cumsum(
				tf.ones(
						shape=(len(action_counts[level - 1]), max_actions[level - 1]),
						dtype=INT_DTYPE,
				),
				exclusive=True,
				name="broadcast_ranges_lvl{}".format(level)
		)
		for level in range(1, levels)
	]
	parents = [
		tf.Variable(
				[np.nan],
				name="parents_lvl0",
		) if level == 0
		else tf.boolean_mask(
				broadcast_ranges[level - 1],
				mask=mask_children[level - 1],
				name="parents_lvl{}".format(level),
		)
		for level in range(levels)
	]
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

	parents_ = get_parents_from_action_counts(action_counts_)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, parents_)