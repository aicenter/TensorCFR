import os
import tensorflow as tf

from src.domains.Domain import Domain


def get_domain_goofspiel6():
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
			'IIGS6_s1_bf_ft.gbt'
	)
	return Domain.init_from_gambit_file(path_to_domain_filename, domain_name="IIGS6_s1_bf_ft_via_gambit")


# TODO: Get rid of `ACTIVATE_FILE` hotfix in "#74 Storage for large files"
ACTIVATE_FILE = False

if __name__ == '__main__' and ACTIVATE_FILE:
	goofspiel6 = get_domain_goofspiel6()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel6.print_domain(sess)
