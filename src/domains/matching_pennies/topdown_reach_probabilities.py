import tensorflow as tf

from src.domains.matching_pennies.domain_definitions import reach_probability_of_root_node, node_to_infoset, \
	current_infoset_strategies, levels, infoset_acting_players
from src.domains.matching_pennies.node_strategies import get_node_cf_strategies
from src.domains.matching_pennies.swap_players import swap_players
from src.utils.tensor_utils import print_tensors, expanded_multiply, scatter_nd_sum


# game of matching pennies: see doc/matching_pennies_efg_illustration.jpg

def get_nodal_reach_probabilities():
	# TODO take into account swapping players
	node_cf_strategies = get_node_cf_strategies()
	nodal_reach_probabilities = [None] * levels
	nodal_reach_probabilities[0] = reach_probability_of_root_node
	for level in range(1, levels):
		nodal_reach_probabilities[level] = expanded_multiply(
				expandable_tensor=nodal_reach_probabilities[level - 1],
				expanded_tensor=node_cf_strategies[level - 1],
				name="nodal_reach_probabilities_lvl{}".format(level))
	return nodal_reach_probabilities


def get_infoset_reach_probabilities():
	nodal_reach_probabilities = get_nodal_reach_probabilities()
	infoset_reach_probabilities = [None] * levels
	infoset_reach_probabilities[0] = tf.identity(nodal_reach_probabilities[0], name="infoset_reach_probabilities_lvl0")
	for level in range(1, levels - 1):
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
