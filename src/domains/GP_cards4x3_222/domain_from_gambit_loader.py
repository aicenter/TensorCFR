import os
os.environ["CUDA_VISIBLE_DEVICES"]="-1"   # off GPU
import tensorflow as tf

from src.domains.Domain import Domain


def get_domain_GP_cards4x3_222():
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
	return Domain.init_from_gambit_file(path_to_domain_filename, domain_name="GP_cards4x3_222_via_gambit")


if __name__ == '__main__':
	goofspiel3 = get_domain_GP_cards4x3_222()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		goofspiel3.print_domain(sess)
