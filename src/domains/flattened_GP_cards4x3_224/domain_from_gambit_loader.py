import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_GP_cards4x3_224():
	path_to_domain_filename = os.path.join(
			os.path.dirname(
					os.path.abspath(
							__file__)
			),
			'..',
			'..',
			'..',
			'doc',
			'poker',
			'GP_cards4x3_224.gbt'
	)
	return FlattenedDomain.init_from_gambit_file(path_to_domain_filename, domain_name="GP_cards4x3_224_gambit_flattened")


if __name__ == '__main__':
	poker = get_flattened_GP_cards4x3_224()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		poker.print_domain(sess)
