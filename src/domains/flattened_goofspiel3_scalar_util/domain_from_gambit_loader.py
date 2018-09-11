import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_goofspiel3_scalar_util():
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
			'II-GS3_scalar_util.efg'
	)
	return FlattenedDomain.init_from_hkl_file(path_to_domain_filename, domain_name="II-GS3_scalar_util_gambit_flattened")


if __name__ == '__main__':
	goofspiel3_scalar_util = get_flattened_goofspiel3_scalar_util()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel3_scalar_util.print_domain(sess)
