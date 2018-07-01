#!/usr/bin/env python3
import numpy as np
import tensorflow as tf


from src.commons.constants import INT_DTYPE, TERMINAL_NODE, INNER_NODE
from pprint import pprint
from src.utils.tensor_utils import print_tensors, print_tensor, scatter_nd_sum


def distribute_strategies_to_nodes(infoset_strategies, node_to_infoset, name, updating_player=None,
                                   acting_players=None):
	"""
  Translate 2-D tensor `infoset_strategies` of strategies per information sets to strategies per game states.
  The translation is done based on N-D tensor `states_to_infosets`: each state (indexed by N-D coordinate)
  stores the index of its information set.

  If both `updating_player` and `acting_players` are `None` (default), no masking is used for strategies. Otherwise,
  the `updating_player` acts with probabilities 1 everywhere (for the reach probability in the formula of
  counterfactual values).

  The corresponding TensorFlow operation (in the computation graph) outputs (N+1)-D tensor, which gives
  for every states (indexed by N-D coordinate) the corresponding strategy of its information set. The strategy
  can be read out in the final (N+1)th dimension.

  Args:
    :param infoset_strategies: A 2-D tensor of floats.
    :param node_to_infoset: An N-D tensor of ints.
    :param name: A string to name the resulting tensor operation.
    :param updating_player: The index of the updating player to create for counterfactual probabilities.
    :param acting_players: A tensor of the same shape as `node_to_infoset`, representing acting players per infosets.

  Returns:
    A corresponding TensorFlow operation (from the computation graph).
  """
	if (updating_player is not None) and (acting_players is not None):  # counterfactual reach probabilities
		strategies = tf.where(
				condition=tf.equal(acting_players, updating_player),
				x=tf.ones_like(infoset_strategies),
				y=infoset_strategies
		)
	else:
		strategies = infoset_strategies
	return tf.gather(params=strategies, indices=node_to_infoset, name=name)


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


def get_node_types_from_action_counts(action_counts):
	levels = len(action_counts)
	return [
		tf.where(
				tf.equal(
						action_counts[level],
						0,
				),
				x=[TERMINAL_NODE] * len(action_counts[level]),
				y=[INNER_NODE] * len(action_counts[level]),
				name="node_types_lvl{}".format(level),
		)
		for level in range(levels)
	]


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