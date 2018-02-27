import tensorflow as tf


def assign_strategies_to_nodes(infoset_strategies, node_to_infoset, name):
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

	Returns:
		A corresponding TensorFlow operation (from the computation graph).
	"""
	return tf.gather(params=infoset_strategies, indices=node_to_infoset, name=name)