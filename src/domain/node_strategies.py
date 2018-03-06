import tensorflow as tf

from assign_strategies_to_nodes import assign_strategies_to_nodes
from constants import PLAYER1
from domain.domain_01 import node_to_IS_lvl0, IS_strategies_lvl0, node_to_IS_lvl1, \
	IS_strategies_lvl1, node_to_IS_lvl2, IS_strategies_lvl2, IS_acting_players_lvl0, IS_acting_players_lvl1, \
	IS_acting_players_lvl2
from utils.tensor_utils import print_tensors


# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)


def get_node_strategies():
	node_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies_lvl0, node_to_IS_lvl0, name="node_strategies_lvl0")
	node_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies_lvl1, node_to_IS_lvl1, name="node_strategies_lvl1")
	node_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies_lvl2, node_to_IS_lvl2, name="node_strategies_lvl2")
	return [node_strategies_lvl0, node_strategies_lvl1, node_strategies_lvl2]


def get_node_cf_strategies(updating_player=PLAYER1):
	# TODO generate node_cf_strategies_* with tf.where on node_strategies
	node_cf_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies_lvl0, node_to_IS_lvl0,
	                                                     updating_player=updating_player,
	                                                     acting_players=IS_acting_players_lvl0,
	                                                     name="node_cf_strategies_lvl0")
	node_cf_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies_lvl1, node_to_IS_lvl1,
	                                                     updating_player=updating_player,
	                                                     acting_players=IS_acting_players_lvl1,
	                                                     name="node_cf_strategies_lvl1")
	node_cf_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies_lvl2, node_to_IS_lvl2,
	                                                     updating_player=updating_player,
	                                                     acting_players=IS_acting_players_lvl2,
	                                                     name="node_cf_strategies_lvl2")
	return [node_cf_strategies_lvl0, node_cf_strategies_lvl1, node_cf_strategies_lvl2]


if __name__ == '__main__':
	updating_player = PLAYER1
	node_strategies = get_node_strategies()
	node_cf_strategies = get_node_cf_strategies(updating_player=updating_player)
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("########## Level 0 ##########")
		print_tensors(sess, [node_to_IS_lvl0, IS_strategies_lvl0, node_strategies[0],
		                     node_cf_strategies[0]])
		print("########## Level 1 ##########")
		print_tensors(sess, [node_to_IS_lvl1, IS_strategies_lvl1, node_strategies[1],
		                     node_cf_strategies[1]])
		print("########## Level 2 ##########")
		print_tensors(sess, [node_to_IS_lvl2, IS_strategies_lvl2, node_strategies[2],
		                     node_cf_strategies[2]])
