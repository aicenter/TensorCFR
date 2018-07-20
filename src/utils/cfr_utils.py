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


def flatten_strategies_via_action_counts(node_strategies, action_counts, basename="nodal_strategies"):
	levels = len(action_counts)
	return [
		tf.constant(
				[REACH_PROBABILITY_OF_ROOT],
				name="flattened_{}_lvl0".format(basename)
		) if level == 0
		else tf.boolean_mask(
				node_strategies[level - 1],
				mask=tf.sequence_mask(action_counts[level - 1]),
				name="flattened_{}_lvl{}".format(basename, level),
		)
		for level in range(levels)
	]


def expand_to_2D_via_action_counts(action_counts, values_in_children, name="2D_cf_values"):
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
			name="final_expanded_{}".format(name)
	)


def get_action_and_infoset_values(values_in_children, action_counts, parental_node_to_infoset, infoset_strategy,
                                  name="cf_values"):
	"""
  Compute counterfactual values of actions and information sets for one level.

  Args:
    :param values_in_children: A 1-D tensor of counterfactual values for the children.
    :param action_counts: A 1-D array containing number of actions of each node.
    :param parental_node_to_infoset: A 1-D array indicating the index of the infoset for each parent of the children.
     This array is for the parental level, i.e. it is the same as `node_to_infoset[parental_level]`.
    :param infoset_strategy: A 2-D representation of probability of playing an action in an infoset.
		:param name: A string to name the resulting tensor operation.

  Returns: A pair of counterfactual values for all actions in infosets (2-D) and the cfvs of infosets expanded to (2-D)
  to be able to subtract for the first return value.
	"""
	values_in_parent_x_action = expand_to_2D_via_action_counts(
			action_counts,
			values_in_children,
			name="parent_x_action_{}".format(name)
	)
	cfvs_infoset_x_action = scatter_nd_sum(
			indices=tf.expand_dims(parental_node_to_infoset, axis=1),
			updates=values_in_parent_x_action,
			shape=tf.shape(infoset_strategy),
			name="infoset_x_action_{}".format(name)
	)
	cfvs_infoset = tf.reduce_sum(
			tf.multiply(cfvs_infoset_x_action, infoset_strategy),
			axis=1,
			keepdims=True,
			name="infoset_{}".format(name)
	)
	return cfvs_infoset_x_action, cfvs_infoset


def distribute_strategies_to_inner_nodes(infoset_strategies, node_to_infoset, mask_of_inner_nodes, name,
                                         updating_player=None, acting_players=None):
	"""
	The same as the function `distribute_strategies_to_inner_nodes()` with the only difference that strategies are
	 distributed just to inner nodes, not to terminal nodes.

  Args:
    :param infoset_strategies: A 2-D tensor of floats.
    :param node_to_infoset: An N-D tensor of ints.
    :param mask_of_inner_nodes: A boolean mask whether the inner node at the corresponding position is an inner node.
    :param name: A string to name the resulting tensor operation.
    :param updating_player: The index of the updating player to create for counterfactual probabilities.
    :param acting_players: A tensor of the same shape as `node_to_infoset`, representing acting players per infosets.

  Returns:
    A corresponding TensorFlow operation (from the computation graph).
  """
	inner_node_to_infoset = tf.boolean_mask(
		node_to_infoset,
		mask=mask_of_inner_nodes,
		name="inner_node_to_infoset_for_{}".format(name)
	)
	return distribute_strategies_to_nodes(
		infoset_strategies=infoset_strategies,
		node_to_infoset=inner_node_to_infoset,
		name=name,
		updating_player=updating_player,
		acting_players=acting_players
	)


if __name__ == '__main__':
	"""
	Demonstrate on `domains.hunger_games`:
	
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
