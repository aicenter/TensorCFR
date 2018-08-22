import tensorflow as tf

from src.commons.constants import SEED_FOR_TESTING
from src.domains.flattened_domain01_gambit.domain_from_gambit_loader import get_flattened_domain01_from_gambit

if __name__ == '__main__':
	flattened_domain01 = get_flattened_domain01_from_gambit()
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		seeds = [None, SEED_FOR_TESTING, 42]
		for i, seed in enumerate(seeds):
			trunk_depth_ = 2
			print("Random strategies #{} (trunk depth {}, seed {}):".format(i + 1, trunk_depth_, seed))
			strategies = flattened_domain01.generate_random_strategies(
				seed=seed,
				trunk_depth=trunk_depth_
			)
			for level_, strategy_ in enumerate(strategies):
				print("Level {}".format(level_))
				print(strategy_)
