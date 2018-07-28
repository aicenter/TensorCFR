import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_goofspiel5():
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
			'IIGS5_s1_bf_ft.gbt'
	)
	return FlattenedDomain.init_from_gambit_file(path_to_domain_filename, domain_name="IIGS5_s1_bf_ft_gambit_flattened")


# TODO: Get rid of `ACTIVATE_FILE` hotfix
ACTIVATE_FILE = False

if __name__ == '__main__' and ACTIVATE_FILE:
	goofspiel5 = get_flattened_goofspiel5()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel5.print_domain(sess)
