#!/usr/bin/env python3

import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.available_domains import get_domain_by_name

if __name__ == '__main__':
	domain01 = get_domain_by_name("domain01")
	matching_pennies = get_domain_by_name("matching_pennies")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
			print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
			# TODO extract following lines to a UnitTest
			tensorcfr.show_expected_values(sess)
			sess.run(tensorcfr.swap_players())
			print("-----------Swap players-----------\n")
			tensorcfr.show_expected_values(sess)
