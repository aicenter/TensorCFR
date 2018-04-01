#!/usr/bin/env python3

import tensorflow as tf

from assign_strategies_to_nodes import assign_strategies_to_nodes
from constants import TERMINAL_NODE
from domains.domain01.domain_01 import node_to_IS, IS_strategies, utilities, node_types
from utils.tensor_utils import print_tensors

# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png

node_strategies = [None] * 3
node_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies[0], node_to_IS[0], name="node_strategies_lvl0")
node_strategies[0] = node_strategies_lvl0
node_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies[1], node_to_IS[1], name="node_strategies_lvl1")
node_strategies[1] = node_strategies_lvl1
node_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies[2], node_to_IS[2], name="node_strategies_lvl2")
node_strategies[2] = node_strategies_lvl2


def get_expected_values():
    expected_values_lvl3 = tf.identity(utilities[3], name="expected_values_lvl3")

    weighted_sum_of_values_lvl2 = tf.reduce_sum(
        input_tensor=node_strategies[2] * expected_values_lvl3,
        axis=-1,
        name="weighted_sum_of_values_lvl2"
    )

    expected_values_lvl2 = tf.where(
        condition=tf.equal(node_types[2], TERMINAL_NODE),
        x=utilities[2],
        y=weighted_sum_of_values_lvl2,
        name="expected_values_lvl2"
    )

    weighted_sum_of_values_lvl1 = tf.reduce_sum(
        input_tensor=node_strategies[1] * expected_values_lvl2,
        axis=-1,
        name="weighted_sum_of_values_lvl1"
    )

    expected_values_lvl1 = tf.where(
        condition=tf.equal(node_types[1], TERMINAL_NODE),
        x=utilities[1],
        y=weighted_sum_of_values_lvl1,
        name="expected_values_lvl1"
    )

    weighted_sum_of_values_lvl0 = tf.reduce_sum(
        input_tensor=node_strategies[0] * expected_values_lvl1,
        axis=-1,
        name="weighted_sum_of_values_lvl0"
    )

    expected_values_lvl0 = tf.where(
        condition=tf.equal(node_types[0], TERMINAL_NODE),
        x=utilities[0],
        y=weighted_sum_of_values_lvl0,
        name="expected_values_lvl0"
    )

    return [expected_values_lvl0, expected_values_lvl1, expected_values_lvl2, expected_values_lvl3]


if __name__ == '__main__':
    expected_values = get_expected_values()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        print("########## Level 3 ##########")
        print_tensors(sess, [utilities[3], expected_values[3]])
        print("########## Level 2 ##########")
        print_tensors(sess, [node_strategies_lvl2])
        print_tensors(sess, [utilities[2], expected_values[2]])
        print("########## Level 1 ##########")
        print_tensors(sess, [node_strategies_lvl1])
        print_tensors(sess, [utilities[1], expected_values[1]])
        print("########## Level 0 ##########")
        print_tensors(sess, [node_strategies_lvl0])
        print_tensors(sess, [utilities[0], expected_values[0]])
