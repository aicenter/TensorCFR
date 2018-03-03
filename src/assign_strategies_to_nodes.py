import tensorflow as tf


def assign_strategies_to_nodes(infoset_strategies, node_to_infoset, name, updating_player=None, acting_players=None):
	"""Translate 2-D tensor `infoset_strategies` of strategies per information sets to strategies per game states.
	The translation is done based on N-D tensor `states_to_infosets`: each state (indexed by N-D coordinate)
	stores the index of its information set.

	The corresponding TensorFlow operation (in the computation graph) outputs (N+1)-D tensor, which gives
	for every states (indexed by N-D coordinate) the corresponding strategy of its information set. The strategy
	can be read out in the final (N+1)th dimension

	Args:
		:param infoset_strategies: A 2-D tensor of floats.
		:param node_to_infoset: An N-D tensor of ints.
		:param name: A string to name the resulting tensor operation.
		:param updating_player: The index of the updating player. If both `updating_player` and `acting_acting_players` are
			`None` (default), no masking is used for strategies.
		:param acting_players: A tensor of the same shape as `node_to_infoset`, representing acting players per infosets.

	Returns:
		A corresponding TensorFlow operation (from the computation graph).
	"""
	if (updating_player is not None) and (acting_players is not None):  # counterfactual reach probabilities
		strategies = tf.where(condition=tf.equal(acting_players, updating_player), x=tf.ones_like(infoset_strategies),
		                      y=infoset_strategies)
	else:
		strategies = infoset_strategies
	return tf.gather(params=strategies, indices=node_to_infoset, name=name)
