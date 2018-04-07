import tensorflow as tf

from src.constants import PLAYER1
from src.domains.domain01.domain_01 import node_to_infoset, infoset_strategies, infoset_acting_players
from src.utils.assign_strategies_to_nodes import assign_strategies_to_nodes
from src.utils.tensor_utils import print_tensors


# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png


def get_node_strategies():
    node_strategies_lvl0 = assign_strategies_to_nodes(infoset_strategies[0], node_to_infoset[0], name="node_strategies_lvl0")
    node_strategies_lvl1 = assign_strategies_to_nodes(infoset_strategies[1], node_to_infoset[1], name="node_strategies_lvl1")
    node_strategies_lvl2 = assign_strategies_to_nodes(infoset_strategies[2], node_to_infoset[2], name="node_strategies_lvl2")
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

