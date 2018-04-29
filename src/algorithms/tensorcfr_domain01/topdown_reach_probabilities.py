import tensorflow as tf

from src.algorithms.tensorcfr_domain01.node_strategies import get_node_cf_strategies
from src.domains.domain01.domain_definitions import reach_probability_of_root_node, node_to_infoset, \
	current_infoset_strategies, levels, infoset_acting_players, current_opponent, current_updating_player
from src.utils.tensor_utils import print_tensors, expanded_multiply, scatter_nd_sum


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

def get_nodal_reach_probabilities(for_player=current_updating_player):
	"""
	:param for_player: The player for which the reach probabilities are computed. These probabilities are usually computed
	 for the updating player when counterfactual values are computed. Therefore, `for_player` is set to
	  `current_updating_player` by default.
	:return: The reach probabilities of nodes based on `current_infoset_strategies`.
	"""
	node_cf_strategies = get_node_cf_strategies()
	with tf.variable_scope("nodal_reach_probabilities"):
		nodal_reach_probabilities = [None] * levels
		nodal_reach_probabilities[0] = reach_probability_of_root_node
		for level in range(1, levels):
			nodal_reach_probabilities[level] = expanded_multiply(
					expandable_tensor=nodal_reach_probabilities[level - 1],
					expanded_tensor=node_cf_strategies[level - 1],
					name="nodal_reach_probabilities_lvl{}".format(level)
			)
		return nodal_reach_probabilities


def get_infoset_reach_probabilities(for_player=current_opponent):
	"""
	:param for_player: The player for which the reach probabilities are computed. These probabilities are usually computed
	 for the opponent when his strategies are cumulated. Therefore, `for_player` is set to `current_opponent` by default.
	:return: The reach probabilities of information sets based on `current_infoset_strategies`.
	"""
	nodal_reach_probabilities = get_nodal_reach_probabilities(for_player)
	with tf.variable_scope("infoset_reach_probabilities"):
		infoset_reach_probabilities = [None] * levels
		with tf.variable_scope("level0"):
			infoset_reach_probabilities[0] = tf.identity(nodal_reach_probabilities[0],
			                                             name="infoset_reach_probabilities_lvl0")
		for level in range(1, levels - 1):
			with tf.variable_scope("level{}".format(level)):
				scatter_nd_sum_indices = tf.expand_dims(
						node_to_infoset[level],
						axis=-1,
						name="expanded_node_to_infoset_lvl{}".format(level))
				scatter_nd_sum_updates = nodal_reach_probabilities[level]
				scatter_nd_sum_shape = infoset_acting_players[level].shape
				infoset_reach_probabilities[level] = scatter_nd_sum(
						indices=scatter_nd_sum_indices,
						updates=scatter_nd_sum_updates,
						shape=scatter_nd_sum_shape,
						name="infoset_reach_probabilities_lvl{}".format(level)
				)
	return infoset_reach_probabilities


def show_reach_probabilities(session):
	for level_ in range(levels):
		print("########## Level {} ##########".format(level_))
		print_tensors(sess, [nodal_reach_probabilities_[level_]])
		if level_ < levels - 1:
			print_tensors(session, [
				node_to_infoset[level_],
				infoset_reach_probabilities_[level_],
				current_infoset_strategies[level_],
				node_cf_strategies_[level_],
			])


if __name__ == '__main__':
	from src.algorithms.tensorcfr_domain01.swap_players import swap_players

	node_cf_strategies_ = get_node_cf_strategies()
	nodal_reach_probabilities_ = get_nodal_reach_probabilities()
	infoset_reach_probabilities_ = get_infoset_reach_probabilities()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# TODO extract following lines to a UnitTest
		show_reach_probabilities(sess)
		print("-----------Swap players-----------\n")
		sess.run(swap_players())
		show_reach_probabilities(sess)
