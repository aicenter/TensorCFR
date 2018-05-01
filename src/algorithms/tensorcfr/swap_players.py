import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import swap_players
from src.commons.constants import PLAYER1, PLAYER2


# custom-made game: see doc/domain01_via_drawing.png and doc/domain01_via_gambit.png


if __name__ == '__main__':
	from src.domains.domain01.domain_definitions import cfr_step, print_misc_variables

	increment_cfr_step_op = tf.assign_add(ref=cfr_step, value=1, name="increment_cfr_step_op")
	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("PLAYER1 is {}, PLAYER2 is {}".format(PLAYER1, PLAYER2))
		for _ in range(5):
			print("########## CFR step {} ##########\n".format(cfr_step.eval()))
			print_misc_variables(session=sess)
			sess.run([swap_players(), increment_cfr_step_op])
