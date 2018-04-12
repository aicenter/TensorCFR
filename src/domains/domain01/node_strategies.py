import tensorflow as tf

from src.constants import PLAYER1
from src.domains.domain01.domain01 import node_to_infoset, infoset_strategies, infoset_acting_players
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

# noinspection PyShadowingNames
def assign_strategies_to_nodes(infoset_strategies, node_to_infoset, name, updating_player=None, acting_players=None):
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
		strategies = tf.where(condition=tf.equal(acting_players, updating_player), x=tf.ones_like(infoset_strategies),
		                      y=infoset_strategies)
	else:
		strategies = infoset_strategies
	return tf.gather(params=strategies, indices=node_to_infoset, name=name)


def get_node_strategies():
	node_strategies_lvl0 = assign_strategies_to_nodes(infoset_strategies[0], node_to_infoset[0],
	                                                  name="node_strategies_lvl0")
	node_strategies_lvl1 = assign_strategies_to_nodes(infoset_strategies[1], node_to_infoset[1],
	                                                  name="node_strategies_lvl1")
	node_strategies_lvl2 = assign_strategies_to_nodes(infoset_strategies[2], node_to_infoset[2],
	                                                  name="node_strategies_lvl2")
	return [node_strategies_lvl0, node_strategies_lvl1, node_strategies_lvl2]


def get_node_cf_strategies(updating_player=PLAYER1):
	# TODO generate node_cf_strategies_* with tf.where on node_strategies
	node_cf_strategies_lvl0 = assign_strategies_to_nodes(
			infoset_strategies[0],
			node_to_infoset[0],
			updating_player=updating_player,
			acting_players=infoset_acting_players[0],
			name="node_cf_strategies_lvl0"
	)
	node_cf_strategies_lvl1 = assign_strategies_to_nodes(
			infoset_strategies[1],
			node_to_infoset[1],
			updating_player=updating_player,
			acting_players=infoset_acting_players[1],
			name="node_cf_strategies_lvl1"
	)
	node_cf_strategies_lvl2 = assign_strategies_to_nodes(
			infoset_strategies[2],
			node_to_infoset[2],
			updating_player=updating_player,
			acting_players=infoset_acting_players[2],
			name="node_cf_strategies_lvl2"
	)
	return [node_cf_strategies_lvl0, node_cf_strategies_lvl1, node_cf_strategies_lvl2]


if __name__ == '__main__':
	updating_player = PLAYER1
	node_strategies = get_node_strategies()
	node_cf_strategies = get_node_cf_strategies(updating_player=updating_player)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Level 0 ##########")
		print_tensors(
				sess,
				[
					node_to_infoset[0],
					infoset_strategies[0],
					node_strategies[0],
					node_cf_strategies[0]
				]
		)
		print("########## Level 1 ##########")
		print_tensors(
				sess,
				[
					node_to_infoset[1],
					infoset_strategies[1],
					node_strategies[1],
					node_cf_strategies[1]
				]
		)
		print("########## Level 2 ##########")
		print_tensors(
				sess,
				[
					node_to_infoset[2],
					infoset_strategies[2],
					node_strategies[2],
					node_cf_strategies[2]
				]
		)
