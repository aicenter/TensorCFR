import os

import tensorflow as tf

from src.domains.FlattenedDomain import FlattenedDomain
from src.utils.other_utils import activate_script


def get_flattened_goofspiel6():
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
	return FlattenedDomain.init_from_hkl_file(path_to_domain_filename, domain_name="IIGS6_s1_bf_ft_gambit_flattened")


if __name__ == '__main__' and activate_script():
	goofspiel6 = get_flattened_goofspiel6()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel6.print_domain(sess)
