#!/usr/bin/env python3
from pprint import pprint

import numpy as np
import tensorflow as tf

from src.commons.constants import INT_DTYPE, TERMINAL_NODE, INNER_NODE, REACH_PROBABILITY_OF_ROOT, FLOAT_DTYPE
from src.utils.tensor_utils import print_tensors, scatter_nd_sum


def distribute_strategies_to_nodes(infoset_strategies, node_to_infoset, name, updating_player=None,
                                   acting_players=None):
	"""
  Distribute 2-D tensor `infoset_strategies` of strategies per information sets to strategies per game nodes.
  The translation is done based on N-D tensor `node_to_infosets`: each node (indexed by N-D coordinate)
  stores the index of its information set.

  If both `updating_player` and `acting_players` are `None` (default), no masking is used for strategies. Otherwise,
  the `updating_player` acts with probabilities 1 everywhere (for the reach probability in the formula of
  counterfactual values).

  The corresponding TensorFlow operation (in the computation graph) outputs (N+1)-D tensor, which gives
  for every node (indexed by N-D coordinate) the corresponding strategy of its information set. The strategy
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


def flatten_via_action_counts(node_strategies, action_counts, basename="node_strategies"):
	levels = len(action_counts)
	return [
		tf.constant(
				REACH_PROBABILITY_OF_ROOT,
				name="flattened_node_strategies_lvl0"
		) if level == 0
		else tf.boolean_mask(
				tf.expand_dims(node_strategies[0], axis=0) if level == 1
				else node_strategies[level - 1],
				mask=tf.sequence_mask(action_counts[level - 1]),
				name="flattened_{}_lvl{}".format(basename, level),
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


def expand_to_2D_via_action_counts(action_counts, values_in_children, name="reshape_CFVs"):
	"""
  Reshape data related to children (e.g., CFVs) to a 2D tensor of shape (parent x action).

  Args:
    :param action_counts: A 1-D array containing number of actions of each node.
    :param values_in_children: Data for the children to reshape.
		:param name: A string to name the resulting tensor operation.

  Returns: A corresponding TensorFlow operation (from the computation graph) that computes the (parent x action)
  tensor with the provided data distributed to the correct positions.
	"""
	mask_children = tf.sequence_mask(
			action_counts,
			name="initial_boolean_mask_in_{}".format(name)
	)
	mask_children_int_dtype = tf.cast(
			mask_children,
			dtype=INT_DTYPE,
			name="initial_integer_mask_in_{}".format(name)
	)
	first_column = tf.expand_dims(
			tf.cumsum(action_counts,
			          exclusive=True,
			          name="first_column_indices_in_{}".format(name)),
			dim=1
	)
	mask_with_replaced_first_column = tf.concat(
			[first_column, mask_children_int_dtype[:, 1:]], 1,
			name="replacing_col0_in_{}".format(name)
	)
	indices_2D_into_1D = tf.expand_dims(
			tf.cumsum(
					mask_with_replaced_first_column,
					axis=1
			),
			dim=2,
			name="computing_indices_in_{}".format(name)
	)
	return tf.where(
			condition=mask_children,
			x=tf.gather_nd(
					values_in_children,
					indices_2D_into_1D
			),
			y=tf.zeros_like(
					mask_children,
					dtype=FLOAT_DTYPE
			),
			name="cleaned_up_result_in_{}".format(name)
	)


def get_action_and_infoset_cf_values(children_values, action_counts, parent_IS_map, strategy):
	"""
  Compute counterfactual values of actions and information sets for one level.

  Args:
    :param children_values: A 1-D tensor of counterfactual values for the children.
    :param action_counts: A 1-D array containing number of actions of each node.
    :param parent_IS_map: A 1-D array indicating the index if the IS for each parent of the children.
    :param strategy: A 2-D representation of probability of playing an action in an IS.

  Returns: A pair of counterfactual values for all actions in ISs (2-D) and the cfvs of ISs expanded to (2-D) to be
  able to subtract for the first return value.
	"""
	parent_x_action = expand_to_2D_via_action_counts(action_counts, children_values)
	cfvs_IS_action = scatter_nd_sum(
			tf.expand_dims(parent_IS_map, axis=1),
			parent_x_action,
			tf.shape(strategy),
			name="sums_cfvs_over_IS"
	)
	cfvs_IS = tf.reduce_sum(
			tf.multiply(cfvs_IS_action, strategy),
			axis=1,
			name="IS_cfvs_from_action_cfvs"
	)
	return cfvs_IS_action, tf.expand_dims(cfvs_IS, dim=1)
