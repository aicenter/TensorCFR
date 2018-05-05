import os
import tensorflow as tf

from src.domains.Domain import Domain


def get_domain_matching_pennies():
	path_to_domain_filename = os.path.join(
			os.path.dirname(
					os.path.abspath(
							__file__)
			),
			'..',
			'..',
			'..',
			'doc',
			'matching_pennies',
			'matching_pennies_via_gambit.efg'
	)
	return Domain.init_from_gambit_file(path_to_domain_filename)


if __name__ == '__main__':
	matching_pennies = get_domain_matching_pennies()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		matching_pennies.print_domain(sess)
		# TODO Why does GambitEFGLoader initialize strategies to:
		# "from_gambit/initial_infoset_strategies_lvl1:0"
    # [[0.25 0.25]]

