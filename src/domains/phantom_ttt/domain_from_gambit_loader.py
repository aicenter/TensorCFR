import os

import tensorflow as tf

from src.commons.constants import PROJECT_ROOT
from src.domains.FlattenedDomain import FlattenedDomain


def get_domain_phantom_ttt_single_level_IS():
	path_to_domain_filename = os.path.join(
		PROJECT_ROOT,
		'doc',
		'phantom_ttt',
		'SingleLevelPhantomTTT.efg'
	)

	return FlattenedDomain.init_from_hkl_file(path_to_domain_filename, domain_name="phantom_ttt_single_level_IS_gambit_flattened")


if __name__ == '__main__':
	phantom_ttt = get_domain_phantom_ttt_single_level_IS()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		phantom_ttt.print_domain(sess)
