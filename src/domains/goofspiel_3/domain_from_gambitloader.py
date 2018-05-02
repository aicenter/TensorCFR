import os
import tensorflow as tf

from src.domains.Domain import Domain

if __name__ == '__main__':
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

	domain_ = Domain.init_from_gambit_file(path_to_domain_filename)

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		domain_.print_domain(sess)
