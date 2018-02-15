#!/usr/bin/env python3

import tensorflow as tf

from utils.tensor_utils import print_tensors

count_of_actions = 3
shape3x3 = [count_of_actions, count_of_actions]
shape3x3x3 = [count_of_actions, count_of_actions, count_of_actions]
acting_players = tf.constant([[1, 0, 2],
                              [1, 1, 0],
                              [2, 0, 2]],
                             name="acting_players")
# -- NOTE: tf.random_uniform DOESN'T WORK WITH LAZY EVALUATION IN TENSORFLOW!!!
# reach_probabilities_without_normalization=tf.random_uniform([count_of_actions, count_of_actions], minval=0, maxval=1)
reach_probabilities_without_normalization = tf.reshape(tf.range(0, 9), shape3x3)
reach_probabilities_lvl_1 = tf.Variable(reach_probabilities_without_normalization \
                                        / tf.reduce_sum(reach_probabilities_without_normalization),
                                        name="reach_probabilities_lvl_1")
resolving_player = 1
mask_players = tf.equal(acting_players, resolving_player, name="mask_players")
mask_actions = tf.stack(values=[mask_players for _ in range(count_of_actions)], axis=-1, name="mask_actions")
strategies = tf.reshape(tf.range(0, 27), shape3x3x3, name="strategies")

# contribution of strategies to the counterfactual probability
strategies_lvl_2 = tf.where(mask_actions, tf.ones_like(strategies), strategies, name="strategies_lvl_2")

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print_tensors(sess, [reach_probabilities_lvl_1, acting_players, mask_players, mask_actions, strategies,
	                     strategies_lvl_2])
