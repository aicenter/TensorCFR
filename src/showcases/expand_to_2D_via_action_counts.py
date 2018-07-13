#!/usr/bin/env python3

from pprint import pprint

import tensorflow as tf

from src.utils.cfr_utils import get_action_and_infoset_cf_values
from src.utils.tensor_utils import print_tensors

if __name__ == '__main__':
	# showcase using the domain `hunger_games`
	action_counts_ = [
		[2],
		[1, 6],
		[4, 0, 0, 0, 0, 0, 0],
		[3, 3, 2, 2],
		[2] * 10,
		[0] * 20
	]
	print("action_counts:")
	pprint(action_counts_, indent=1, width=80)
	l3children = [31.1, 32, 33, 34, 35, 36, 37, 38, 39, 40]
	parent_IS_map = [0, 0, 1, 1]
	strategy = [[0.1, 0.3, 0.6], [.2, .8, .0]]
	out = get_action_and_infoset_cf_values(l3children, action_counts_[3], parent_IS_map, tf.Variable(strategy))
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print_tensors(sess, out)
