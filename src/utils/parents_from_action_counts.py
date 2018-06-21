#!/usr/bin/env python3
import numpy as np
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
	from pprint import pprint
	from src.utils.tensor_utils import print_tensors

	sizes = [1] + list(map(np.sum, action_counts))   # the 1st size is `1` because there's only 1 root
	print("sizes:")
	pprint(sizes, indent=1, width=50)

	parents = [
		tf.zeros(
				shape=[sizes[level]],
				name="parents_lvl{}".format(level),
		)
		for level in range(len(sizes))
	]

	leftmost_child = [
		tf.cumsum(
				action_counts[level],
				exclusive=False,
				name="leftmost_child_lvl{}".format(level)
		)
		for level in range(len(action_counts))
	]
	with tf.Session() as tmp_sess:
		tmp_sess.run(tf.global_variables_initializer())
		print_tensors(tmp_sess, leftmost_child)

	return parents


if __name__ == '__main__':
	""" Demonstrate on `domains.hunger_games`:
	
	```
	action_counts:
	[[2],
	 [1, 6],
	 [4],
	 [3, 3, 2, 2],
	 [2, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
	```
	 
	expected `parents`:
	```
	"hunger_games/parents_lvl0:0"
	[0]

	"hunger_games/parents_lvl1:0"
	[0 0]

	"hunger_games/parents_lvl2:0"
	[0]

	"hunger_games/parents_lvl3:0"
	[0 0 0 0]

	"hunger_games/parents_lvl4:0"
	[0 0 0 0 0 0 0 0 0 0]
	```
	"""

	action_counts = [
		[2],
		[1, 6],
		[4, 0, 0, 0, 0, 0, 0],
		[3, 3, 2, 2],
		[2] * 10,
		[0] * 20
	]
	parents_ = get_parents_from_action_counts(action_counts)

	from pprint import pprint
	from src.utils.tensor_utils import print_tensors
	print("action_counts:")
	pprint(action_counts, indent=1, width=50)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, parents_)
