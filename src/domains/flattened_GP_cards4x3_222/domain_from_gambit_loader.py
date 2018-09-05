import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_GP_cards4x3_222():
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
			'GP_cards4x3_222.gbt'
	)
	return FlattenedDomain.init_from_hkl_file(path_to_domain_filename, domain_name="GP_cards4x3_222_gambit_flattened")


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False


if __name__ == '__main__' and ACTIVATE_FILE:
	poker = get_flattened_GP_cards4x3_222()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		poker.print_domain(sess)
