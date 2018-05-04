import tensorflow as tf

from src.algorithms.tensorcfr.TensorCFR import TensorCFR
from src.domains.available_domains import get_domain_by_name
from src.utils.tensor_utils import print_tensors

if __name__ == '__main__':
	domain01 = get_domain_by_name("domain01")
	matching_pennies = get_domain_by_name("matching_pennies")
	for tensorcfr in [TensorCFR(domain01), TensorCFR(matching_pennies)]:
		infoset_acting_players = tensorcfr.domain.get_infoset_acting_players()
		infoset_strategies_matched_to_regrets = tensorcfr.get_strategy_matched_to_regrets()
		infoset_reach_probabilities = tensorcfr.get_infoset_reach_probabilities()
		update_infoset_strategies = tensorcfr.update_strategy_of_updating_player()
		ops_cumulate_infoset_strategies = tensorcfr.cumulate_strategy_of_opponent()
		with tf.Session() as sess:
			sess.run(tf.global_variables_initializer())
			print(">>>>>>>>>> {} <<<<<<<<<<".format(tensorcfr.domain.domain_name))
			sess.run(tf.assign(ref=tensorcfr.domain.averaging_delay, value=0))
			print("########## Update strategy ##########")
			for level in range(tensorcfr.domain.acting_depth):
				print("########## Level {} ##########".format(level))
				print_tensors(sess, [
					tensorcfr.domain.current_infoset_strategies[level],
					infoset_acting_players[level],
					tensorcfr.domain.current_updating_player,
					infoset_strategies_matched_to_regrets[level],
					update_infoset_strategies[level],
					tensorcfr.domain.current_infoset_strategies[level]
				])
			print("########## Cumulate strategy ##########")
			for level in range(tensorcfr.domain.acting_depth):
				print("########## Level {} ##########".format(level))
				# TODO add to the unittest of ops_cumulate_infoset_strategies()
				print_tensors(sess, [
					infoset_acting_players[level],
					tensorcfr.domain.current_opponent,
					infoset_reach_probabilities[level],
					tensorcfr.domain.current_infoset_strategies[level],
				])
				for _ in range(3):
					print_tensors(sess, [
						tensorcfr.domain.cfr_step,
						tensorcfr.domain.cumulative_infoset_strategies[level],
						ops_cumulate_infoset_strategies[level],
						tensorcfr.domain.cumulative_infoset_strategies[level],
					])
					# simulate increasing `crf_step`
					sess.run(tf.assign_add(ref=tensorcfr.domain.cfr_step, value=1, name="increment_cfr_step"))
			print("########## Average strategy ##########")
			print_tensors(sess, tensorcfr.get_average_infoset_strategies())
