import tensorflow as tf

from assign_strategies_to_nodes import assign_strategies_to_nodes
from constants import UPDATING_PLAYER, TERMINAL_NODE
from domain.domain_01 import node_to_IS_lvl0, IS_strategies_lvl0, node_to_IS_lvl1, \
	IS_strategies_lvl1, node_to_IS_lvl2, IS_strategies_lvl2, utilities_lvl3, utilities_lvl2, node_types_lvl2, \
	utilities_lvl1, node_types_lvl1, node_types_lvl0, utilities_lvl0
from utils.tensor_utils import print_tensors

# custom-made game: doc/domain_01.png (https://gitlab.com/beyond-deepstack/TensorCFR/blob/master/doc/domain_01.png)

updating_player = UPDATING_PLAYER

node_strategies_lvl0 = assign_strategies_to_nodes(IS_strategies_lvl0, node_to_IS_lvl0, name="node_strategies_lvl0")
node_strategies_lvl1 = assign_strategies_to_nodes(IS_strategies_lvl1, node_to_IS_lvl1, name="node_strategies_lvl1")
node_strategies_lvl2 = assign_strategies_to_nodes(IS_strategies_lvl2, node_to_IS_lvl2, name="node_strategies_lvl2")

expected_values_lvl3 = tf.identity(utilities_lvl3, name="expected_values_lvl3")
weighted_sum_of_values_lvl2 = tf.reduce_sum(input_tensor=node_strategies_lvl2 * expected_values_lvl3, axis=-1,
                                            name="weighted_sum_of_values_lvl2")
expected_values_lvl2 = tf.where(condition=tf.equal(node_types_lvl2, TERMINAL_NODE), x=utilities_lvl2,
                                y=weighted_sum_of_values_lvl2, name="expected_values_lvl2")
weighted_sum_of_values_lvl1 = tf.reduce_sum(input_tensor=node_strategies_lvl1 * expected_values_lvl2, axis=-1,
                                            name="weighted_sum_of_values_lvl1")
expected_values_lvl1 = tf.where(condition=tf.equal(node_types_lvl1, TERMINAL_NODE), x=utilities_lvl1,
                                y=weighted_sum_of_values_lvl1, name="expected_values_lvl1")
weighted_sum_of_values_lvl0 = tf.reduce_sum(input_tensor=node_strategies_lvl0 * expected_values_lvl1, axis=-1,
                                            name="weighted_sum_of_values_lvl0")
expected_values_lvl0 = tf.where(condition=tf.equal(node_types_lvl0, TERMINAL_NODE), x=utilities_lvl0,
                                y=weighted_sum_of_values_lvl0, name="expected_values_lvl0")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("########## Level 3 ##########")
	print_tensors(sess, [utilities_lvl3, expected_values_lvl3])
	print("########## Level 2 ##########")
	print_tensors(sess, [node_strategies_lvl2])
	print_tensors(sess, [utilities_lvl2, expected_values_lvl2])
	print("########## Level 1 ##########")
	print_tensors(sess, [node_strategies_lvl1])
	print_tensors(sess, [utilities_lvl1, expected_values_lvl1])
	print("########## Level 0 ##########")
	print_tensors(sess, [node_strategies_lvl0])
	print_tensors(sess, [utilities_lvl0, expected_values_lvl0])
