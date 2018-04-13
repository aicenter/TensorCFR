import tensorflow as tf

from src.domains.domain01.cfr_step import do_cfr_step
from src.domains.domain01.domain01 import cfr_step, immediate_infoset_strategies, acting_depth, \
	cumulative_infoset_strategies, infosets_of_non_chance_player
from src.domains.domain01.uniform_strategies import get_infoset_uniform_strategies, assign_uniform_strategies_to_players
from src.utils.tensor_utils import print_tensors, normalize

# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png


# TODO extract to a method with `total_steps` as a parameter
total_steps = 50
cfr_step_op = do_cfr_step()
uniform_strategies = get_infoset_uniform_strategies()

with tf.Session() as sess:
	sess.run(tf.global_variables_initializer())

	print("Initializing strategies to uniform ones...\n")
	sess.run(assign_uniform_strategies_to_players())
	print_tensors(sess, immediate_infoset_strategies)

	print("Running {} CFR+ iterations...\n".format(total_steps))
	for _ in range(total_steps):
		print("########## CFR+ step #{} ##########".format(cfr_step.eval()))
		sess.run(cfr_step_op)
		print_tensors(sess, cumulative_infoset_strategies)

	average_infoset_strategies = [
		tf.where(
			condition=infosets_of_non_chance_player[level],
			x=normalize(cumulative_infoset_strategies[level]),
			y=immediate_infoset_strategies[level],
			name="average_infoset_strategies_lvl{}".format(level)
		)
		for level in range(acting_depth)
	]
	print_tensors(sess, average_infoset_strategies)
