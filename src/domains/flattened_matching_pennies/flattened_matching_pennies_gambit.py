import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_matching_pennies_from_gambit():
	path_to_domain_filename = os.path.join(
		os.path.dirname(
			os.path.abspath(
				__file__)
		),
		'..',
		'..',
		'..',
		'doc',
		'matching_pennies',
		'matching_pennies_via_gambit.efg'
	)
	return FlattenedDomain.init_from_gambit_file(path_to_domain_filename, domain_name="flattened_matching_pennies_gambit")


if __name__ == '__main__':
	flattened_matching_pennies = get_flattened_matching_pennies_from_gambit()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		flattened_matching_pennies.print_domain(sess)
