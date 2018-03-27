#!/usr/bin/env python3

import tensorflow as tf

from domains.domain01.bottomup_expected_values import get_expected_values
from domains.domain01.domain_01 import levels, is_strategies, node_to_is, cf_values_is_actions
from domains.domain01.topdown_reach_probabilities import get_reach_probabilities
from utils.tensor_utils import print_tensors


# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png

def get_cf_values_nodes():  # TODO verify and write a unittest
    expected_values = get_expected_values()
    reach_probabilities = get_reach_probabilities()
    cf_values_of_nodes = [tf.multiply(reach_probabilities[level], expected_values[level],
                                      name="node_cf_val_lvl{}".format(level)) for level in range(levels)]
    return cf_values_of_nodes


# noinspection PyPep8Naming
def get_cf_values_IS_actions():  # TODO verify and write a unittest
    node_cf_values = get_cf_values_nodes()
    new_cf_values_IS_action = [None] * (levels - 1)
    new_cf_values_IS_action[0] = tf.assign(
        ref=cf_values_is_actions[0],
        value=tf.expand_dims(node_cf_values[1], axis=0)
    )
    for level in range(1, levels - 1):  # TODO replace for-loop with parallel_map on TensorArray?
        scatter_nd_add_ref = tf.Variable(tf.zeros_like(is_strategies[level]))
        scatter_nd_add_indices = tf.expand_dims(node_to_is[level], axis=-1)
        scatter_nd_add_updates = node_cf_values[level + 1]
        new_cf_values_IS_action[level] = tf.scatter_nd_add(
            ref=scatter_nd_add_ref,
            indices=scatter_nd_add_indices,
            updates=scatter_nd_add_updates
        )

    return [tf.assign(ref=cf_values_is_actions[level], value=new_cf_values_IS_action[level],
                      name="assign_new_cfv_IS_action_lvl{}".format(level)) for level in range(levels - 1)]


# noinspection PyPep8Naming
def get_cf_values_IS():  # TODO verify and write a unittest
    return [tf.expand_dims(tf.reduce_sum(is_strategies[level] * cf_values_is_actions[level], axis=-1), axis=-1,
                           name="cf_values_IS_lvl{}".format(level))
            for level in range(levels - 1)]


if __name__ == '__main__':
    reach_probabilities_ = get_reach_probabilities()
    expected_values_ = get_expected_values()
    cf_values_nodes_ = get_cf_values_nodes()
    IS_strategies_ = is_strategies
    cf_values_IS_actions_ = get_cf_values_IS_actions()
    cf_values_IS_ = get_cf_values_IS()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(levels):
            print("########## Level {} ##########".format(i))

            print_tensors(sess, [reach_probabilities_[i], expected_values_[i], cf_values_nodes_[i]])

            if i < levels - 1:
                print_tensors(sess, [IS_strategies_[i], cf_values_IS_actions_[i], cf_values_is_actions[i], cf_values_IS_[i]])
                # TODO unittest for multiple call of `cf_values_IS` and `cf_values_IS_actions` as below:
                #  print_tensors(sess, [cf_values_IS_actions[i], cf_values_IS_actions_[i], cf_values_IS_actions[i]])

