import os
import tensorflow as tf

from src.domains.Domain import Domain


def get_domain_goofspiel3():
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
			'II-GS3.gbt'
	)
	return Domain.init_from_gambit_file(path_to_domain_filename, domain_name="II-GS3_via_gambit")


if __name__ == '__main__':
	goofspiel_3 = get_domain_goofspiel3()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel_3.print_domain(sess)
