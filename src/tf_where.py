#!/usr/bin/env python3

import tensorflow as tf

count_of_actions = 3
acting_players = tf.constant([[1, 0, 2],
                              [1, 1, 0],
                              [2, 0, 2]])

# -- NOTE: tf.random_uniform DOESN'T WORK WITH LAZY EVALUATION IN TENSORFLOW!!!
# reach_probabilities_without_normalization = tf.random_uniform([count_of_actions, count_of_actions], minval=0, maxval=1)

# reach_probabilities_without_normalization = [[0.05695065, 0.18175821, 0.19880699],
#                                              [0.08976204, 0.15841936, 0.00814814],
#                                              [0.03140438, 0.10418354, 0.17056671]]
reach_probabilities_without_normalization = acting_players
reach_probabilities = reach_probabilities_without_normalization\
                      / tf.reduce_sum(reach_probabilities_without_normalization)
resolving_player = 1

mask_players = tf.equal(acting_players, resolving_player)
shape3x3 = [count_of_actions, count_of_actions, count_of_actions]
strategies = tf.reshape(tf.range(0, 27), shape3x3)

mask_actions = tf.stack(values=[mask_players for _ in range(count_of_actions)], axis=-1)

# reach_probabilities_lvl_2 = tf.where(mask_actions, tf.ones_like(reach_probabilities), reach_probabilities)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print("reach_probabilities:\n", sess.run(reach_probabilities))
	print("mask_players:\n", sess.run(mask_players))
	print("mask_actions:\n", sess.run(mask_actions))
	# print("strategies:\n", sess.run(strategies))
	# print("reach_probabilities_lvl_2:\n", sess.run(reach_probabilities_lvl_2))
