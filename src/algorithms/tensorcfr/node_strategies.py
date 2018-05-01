import tensorflow as tf

from src.domains.domain01.domain_definitions import node_to_infoset, current_infoset_strategies, \
	infoset_acting_players, acting_depth, current_updating_player
from src.utils.distribute_strategies_to_nodes import assign_strategies_to_nodes
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png

# noinspection PyShadowingNames


def get_node_strategies():
	with tf.variable_scope("node_strategies"):
		return [
			assign_strategies_to_nodes(
					current_infoset_strategies[level],
					node_to_infoset[level],
					name="node_strategies_lvl{}".format(level)
			) for level in range(acting_depth)
		]


def get_node_cf_strategies(updating_player=current_updating_player):
	with tf.variable_scope("node_cf_strategies"):
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


if __name__ == '__main__':
	from src.algorithms.tensorcfr_domain01.swap_players import swap_players
	node_strategies = get_node_strategies()
	node_cf_strategies = get_node_cf_strategies()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		# TODO extract following lines to a UnitTest
		show_strategies(session=sess)
		print("-----------Swap players-----------\n")
		sess.run(swap_players())
		show_strategies(session=sess)
