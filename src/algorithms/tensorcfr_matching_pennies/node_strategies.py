import tensorflow as tf

from src.domains.matching_pennies.domain_definitions import node_to_infoset, current_infoset_strategies, \
	infoset_acting_players, acting_depth, current_updating_player
from src.utils.tensor_utils import print_tensors


# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

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
	return [
		assign_strategies_to_nodes(
				current_infoset_strategies[level],
				node_to_infoset[level],
				name="node_strategies_lvl{}".format(level)
		) for level in range(acting_depth)
	]


def get_node_cf_strategies(updating_player=current_updating_player):
	# TODO generate node_cf_strategies_* with tf.where on node_strategies
	return [
		assign_strategies_to_nodes(
				current_infoset_strategies[level],
				node_to_infoset[level],
				updating_player=updating_player,
				acting_players=infoset_acting_players[level],
				name="node_cf_strategies_lvl{}".format(level)
		) for level in range(acting_depth)
	]


def show_strategies(session):
	for level_ in range(acting_depth):
		print("########## Level {} ##########".format(level_))
		print_tensors(session, [
			node_to_infoset[level_],
			current_infoset_strategies[level_],
			node_strategies[level_],
			infoset_acting_players[level_],
			node_cf_strategies[level_],
		])


ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
	from src.algorithms.tensorcfr_matching_pennies.swap_players import swap_players
	node_strategies = get_node_strategies()
	node_cf_strategies = get_node_cf_strategies()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# TODO extract following lines to a UnitTest
		show_strategies(session=sess)
		print("-----------Swap players-----------\n")
		sess.run(swap_players())
		show_strategies(session=sess)
