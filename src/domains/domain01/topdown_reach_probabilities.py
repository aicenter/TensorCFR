import tensorflow as tf

from src.constants import PLAYER1
from src.domains.domain01.domain_01 import reach_probabilities_, node_to_infoset, infoset_strategies, levels
from src.domains.domain01.node_strategies import get_node_cf_strategies
from src.utils.tensor_utils import print_tensors, expanded_multiply


# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png


def get_reach_probabilities():
	node_cf_strategies = get_node_cf_strategies()
	reach_probabilities = [None] * levels
	reach_probabilities[0] = reach_probabilities[0]
	for level in range(1, levels):
		reach_probabilities[level] = expanded_multiply(
				expandable_tensor=reach_probabilities[level - 1],
				expanded_tensor=node_cf_strategies[level - 1],
				name="reach_probabilities_lvl{}".format(level))
	return reach_probabilities


if __name__ == '__main__':
	updating_player = PLAYER1
	node_cf_strategies_ = get_node_cf_strategies(updating_player=updating_player)
	reach_probabilities_ = get_reach_probabilities()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Level 0 ##########")
		print_tensors(sess, [reach_probabilities_[0], node_to_infoset[0], infoset_strategies[0], node_cf_strategies_[0]])
		print("########## Level 1 ##########")
		print_tensors(sess, [reach_probabilities_[1], node_to_infoset[1], infoset_strategies[1], node_cf_strategies_[1]])
		print("########## Level 2 ##########")
		print_tensors(sess, [reach_probabilities_[2], node_to_infoset[2], infoset_strategies[2], node_cf_strategies_[2]])
		print("########## Level 3 ##########")
		print_tensors(sess, [reach_probabilities_[3]])
