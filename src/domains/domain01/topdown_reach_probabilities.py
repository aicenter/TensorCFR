import tensorflow as tf

from src.constants import PLAYER1
from src.domains.domain01.domain01 import reach_probability_of_root_node, node_to_infoset, infoset_strategies, levels, \
	infoset_acting_players
from src.domains.domain01.node_strategies import get_node_cf_strategies
from src.utils.tensor_utils import print_tensors, expanded_multiply, scatter_nd_sum


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png


def get_nodal_reach_probabilities():
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


if __name__ == '__main__':
	updating_player = PLAYER1
	node_cf_strategies_ = get_node_cf_strategies(updating_player=updating_player)
	nodal_reach_probabilities_ = get_nodal_reach_probabilities()
	infoset_reach_probabilities_ = get_infoset_reach_probabilities()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for level_ in range(levels):
			print("########## Level {} ##########".format(level_))
			print_tensors(sess, [nodal_reach_probabilities_[level_]])
			if level_ < levels - 1:
				print_tensors(sess, [
					node_to_infoset[level_],
					infoset_reach_probabilities_[level_],
					infoset_strategies[level_],
					node_cf_strategies_[level_],
				])
