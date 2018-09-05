import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_domain01_from_gambit():
	path_to_domain_filename = os.path.join(
			os.path.dirname(
					os.path.abspath(
							__file__)
			),
			'..',
			'..',
			'..',
			'doc',
			'domain01_via_gambit.efg'
	)
	return FlattenedDomain.init_from_hkl_file(path_to_domain_filename, domain_name="flattened_domain01_gambit")


if __name__ == '__main__':
	flattened_domain01 = get_flattened_domain01_from_gambit()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		flattened_domain01.print_domain(sess)
