#!/usr/bin/env python3

import tensorflow as tf


# TODO move to `utils`
def get_parents_from_action_counts(action_counts):
	"""
	Compute tensor `parents` containing the indices to each node's parent in the previous level.

  Args:
    :param action_counts: A 2-D numpy array containing number of actions of each node.

  Returns:
    A corresponding TensorFlow operation (from the computation graph) that contain the index to each node's parent
     (in the level above).
  """
	# TODO add final level!
	parents = [
		tf.zeros_like(
				action_counts[level],
				name="parents_lvl{}".format(level)
		)
		for level in range(len(action_counts))
	]
	return parents


if __name__ == '__main__':
	pass
