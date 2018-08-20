import os

import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING
from src.domains.FlattenedDomain import FlattenedDomain


def get_flattened_goofspiel3():
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
			'II-GS3.efg'
	)
	return FlattenedDomain.init_from_gambit_file(path_to_domain_filename, domain_name="II-GS3_gambit_flattened")


if __name__ == '__main__':
	goofspiel3 = get_flattened_goofspiel3()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		seeds = [None, SEED_FOR_TESTING, SEED_FOR_TESTING + 1, SEED_FOR_TESTING + 2, 42]
		for i, seed in enumerate(seeds):
			trunk_depth_ = 3
			print("Random strategies #{} (trunk depth {}, seed {}):".format(i + 1, trunk_depth_, seed))
			strategies = goofspiel3.generate_random_strategies(
				seed=seed,
				trunk_depth=trunk_depth_
			)
			for level_, strategy_ in enumerate(strategies):
				print("Level {}".format(level_))
				print(strategy_)
