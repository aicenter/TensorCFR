import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.commons.constants import PLAYER1, PLAYER2
from src.domains.domain01.Domain01 import domain01
from src.domains.matching_pennies.MatchingPennies import matching_pennies

if __name__ == '__main__':
	for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
		cfr_step = tensorcfr.domain.cfr_step
		increment_cfr_step_op = tf.assign_add(ref=cfr_step, value=1, name="increment_cfr_step_op")
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
			print("PLAYER1 is {}, PLAYER2 is {}".format(PLAYER1, PLAYER2))
			for _ in range(5):
				print("########## CFR step {} ##########\n".format(cfr_step.eval()))
				tensorcfr.domain.print_misc_variables(session=sess)
				sess.run([tensorcfr.swap_players(), increment_cfr_step_op])
