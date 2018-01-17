#!/usr/bin/env python3

import tensorflow as tf

count_of_actions = 4
# tensor_of_strategies = tf.Variable([0.0, 1.0, 2.0, 1.0])
tensor_of_strategies = tf.random_uniform([count_of_actions], minval=0, maxval=1)
tensor_of_strategies = tensor_of_strategies / tf.reduce_sum(tensor_of_strategies)
tensor_of_players = tf.Variable([1, 0, 1, 2])
resolving_player = 1

mask_of_players = tf.equal(tensor_of_players, resolving_player)
tensor_of_next_move_strategies = tf.stack([tensor_of_strategies, tensor_of_strategies, tensor_of_strategies,
                                           tensor_of_strategies])
tensor_of_next_move_strategies = tf.where(mask_of_players, tf.ones_like(tensor_of_next_move_strategies),
                                          tensor_of_next_move_strategies)

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())
	print(sess.run(tensor_of_strategies))
	print(sess.run(mask_of_players))
	print(sess.run(tensor_of_next_move_strategies))
