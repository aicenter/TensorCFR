import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_goofspiel2_2cards():
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
			'II-GS2.efg'      # TODO change here to `*_2cards.efg`
	)
	return FlattenedDomain.init_from_gambit_file(path_to_domain_filename, domain_name="II-GS2_gambit_flattened")


if __name__ == '__main__':
	goofspiel2 = get_flattened_goofspiel2_2cards()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel2.print_domain(sess)
