import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.domain01.Domain01 import domain01
from src.domains.matching_pennies.MatchingPennies import matching_pennies
from src.utils.tensor_utils import print_tensors


if __name__ == '__main__':
	total_steps = 4
	for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
		infoset_acting_players_ = tensorcfr.domain.get_infoset_acting_players()
		cfr_step_op = tensorcfr.do_cfr_step()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
			sess.run(tf.assign(ref=tensorcfr.domain.averaging_delay, value=0))
			print("Running {} CFR+ iterations...\n".format(total_steps))
			for _ in range(total_steps):
				print("########## Start of CFR+ step {} ##########".format(tensorcfr.domain.cfr_step.eval()))
				print_tensors(sess, [tensorcfr.domain.current_updating_player, tensorcfr.domain.current_opponent])
				print("___________________________________\n")
				print_tensors(sess, tensorcfr.domain.positive_cumulative_regrets)
				print("___________________________________\n")
				print_tensors(sess, tensorcfr.domain.current_infoset_strategies)
				print("___________________________________\n")
				print_tensors(sess, tensorcfr.domain.cumulative_infoset_strategies)
				sess.run(cfr_step_op)
