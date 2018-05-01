import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.commons.constants import PLAYER1, PLAYER2
from src.domains.domain01.Domain01 import domain01

if __name__ == '__main__':
	tensorcfr_domain01 = TensorCFR(domain01)

	domain01_cfr_step = tensorcfr_domain01.domain.cfr_step
	increment_cfr_step_op = tf.assign_add(ref=domain01_cfr_step, value=1, name="increment_cfr_step_op")

	with tf.Session() as sess:
		sess.run(tf.global_variables_initializer())
		print("PLAYER1 is {}, PLAYER2 is {}".format(PLAYER1, PLAYER2))
		for _ in range(5):
			print("########## CFR step {} ##########\n".format(domain01_cfr_step.eval()))
			tensorcfr_domain01.domain.print_misc_variables(session=sess)
			sess.run([tensorcfr_domain01.swap_players(), increment_cfr_step_op])
