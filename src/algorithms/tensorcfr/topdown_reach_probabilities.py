#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domains = [
		get_domain_by_name("domain01"),
		get_domain_by_name("matching_pennies"),
		get_domain_by_name("hunger_games")
	]
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for tensorcfr in map(TensorCFR, domains):
			print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
			# TODO extract following lines to a UnitTest
			tensorcfr.show_reach_probabilities(sess)
			print("-----------Swap players-----------\n")
			sess.run(tensorcfr.swap_players())
			tensorcfr.show_reach_probabilities(sess)
