import tensorflow as tf

from constants import PLAYER1
from domain.domain_01 import reach_probabilities_lvl0, node_to_IS_lvl0, IS_strategies_lvl0, node_to_IS_lvl1, \
	IS_strategies_lvl1, node_to_IS_lvl2, IS_strategies_lvl2
from domain.node_strategies import get_node_cf_strategies
from utils.tensor_utils import print_tensors, expanded_multiply


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

def get_reach_probabilities():
	_node_cf_strategies = get_node_cf_strategies()
	reach_probabilities_lvl1 = expanded_multiply(expandable_tensor=reach_probabilities_lvl0,
	                                             expanded_tensor=_node_cf_strategies[0], name="reach_probabilities_lvl1")
	reach_probabilities_lvl2 = expanded_multiply(expandable_tensor=reach_probabilities_lvl1,
	                                             expanded_tensor=_node_cf_strategies[1], name="reach_probabilities_lvl2")
	reach_probabilities_lvl3 = expanded_multiply(expandable_tensor=reach_probabilities_lvl2,
	                                             expanded_tensor=_node_cf_strategies[2], name="reach_probabilities_lvl3")
	return [reach_probabilities_lvl0, reach_probabilities_lvl1, reach_probabilities_lvl2, reach_probabilities_lvl3]


if __name__ == '__main__':
	updating_player = PLAYER1
	node_cf_strategies = get_node_cf_strategies(updating_player=updating_player)
	reach_probabilities = get_reach_probabilities()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Level 0 ##########")
		print_tensors(sess, [reach_probabilities[0], node_to_IS_lvl0, IS_strategies_lvl0, node_cf_strategies[0]])
		print("########## Level 1 ##########")
		print_tensors(sess, [reach_probabilities[1], node_to_IS_lvl1, IS_strategies_lvl1, node_cf_strategies[1]])
		print("########## Level 2 ##########")
		print_tensors(sess, [reach_probabilities[2], node_to_IS_lvl2, IS_strategies_lvl2, node_cf_strategies[2]])
		print("########## Level 3 ##########")
		print_tensors(sess, [reach_probabilities[3]])
