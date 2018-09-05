import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_goofspiel3():
	path_to_domain_filename = os.path.join(
			os.path.dirname(
					os.path.abspath(
							__file__)
			),
			'..',
			'..',
			'..',
			'doc',
			'goofspiel',
			'II-GS3.efg'
	)
	return FlattenedDomain.init_from_hkl_file(path_to_domain_filename, domain_name="II-GS3_gambit_flattened")


if __name__ == '__main__':
	goofspiel3 = get_flattened_goofspiel3()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel3.print_domain(sess)
