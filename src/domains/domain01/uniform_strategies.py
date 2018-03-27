import tensorflow as tf

from constants import IMAGINARY_NODE
from domains.domain01.domain_01 import levels, node_types, node_to_is, is_strategies
from utils.tensor_utils import print_tensors


# custom-made game: see doc/domain_01_via_drawing.png and doc/domain_01_via_gambit.png


def get_is_children_types():  # TODO unittest
    is_children_types = [None] * (levels - 1)
    for level in range(levels - 1):
        if level == 0:
            is_children_types[0] = tf.expand_dims(node_types[1], axis=0, name="IS_children_types_lvl0")
        else:
            is_children_types[level] = tf.scatter_nd_update(
                ref=tf.Variable(
                    tf.zeros_like(
                        is_strategies[level],
                        dtype=node_types[level + 1].dtype
                    )
                ),
                indices=tf.expand_dims(node_to_is[level], axis=-1),
                updates=node_types[level + 1],
                name="IS_children_types_lvl{}".format(level)
            )
    return is_children_types


def get_is_uniform_strategies():  # TODO unittest
    is_children_types = get_is_children_types()
    is_uniform_strategies = [None] * (levels - 1)

    for level in range(levels - 1):
        is_uniform_strategies[level] = tf.to_float(tf.not_equal(is_children_types[level], IMAGINARY_NODE))
        # Note: An all-0's row cannot be normalized. This is caused when IS has only imaginary children. As of now,
        #  `tf.divide` produces `nan` in the entire row.
        is_uniform_strategies[level] = tf.divide(
            is_uniform_strategies[level],
            tf.reduce_sum(is_uniform_strategies[level], axis=-1, keep_dims=True),
            name="IS_uniform_strategies_lvl{}".format(level)
        )

    return is_uniform_strategies


if __name__ == '__main__':
    is_uniform_strategies_ = get_is_uniform_strategies()
    is_children_types_ = get_is_children_types()

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for i in range(levels - 1):
            print("########## Level {} ##########".format(i))
            print_tensors(sess, [node_types[i], is_children_types_[i], is_uniform_strategies_[i]])
