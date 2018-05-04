import os
import tensorflow as tf

from src.domains.Domain import Domain


def get_domain01_from_gambit():
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
	return Domain.init_from_gambit_file(path_to_domain_filename)


if __name__ == '__main__':
	domain01_from_gambit = get_domain01_from_gambit()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		domain01_from_gambit.print_domain(sess)
